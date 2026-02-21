
import sqlite3
import struct
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import numpy as np
from typing import List, Dict, Tuple
import threading
import config as config
from crisis_classifier import get_crisis_classifier
from crisis_resources import build_crisis_response

class RAGSystem:
    """
    Complete RAG pipeline for mental health support conversations
    Optimized with GGUF quantization and sqlite-vec
    THREAD-SAFE: Creates separate DB connections per thread
    """
    
    def __init__(self, lightweight_classifier: bool = False):
        print("[RAGSystem] Initializing OPTIMIZED RAG pipeline...")
        
        # Initialize embedding model
        print(f"[RAGSystem] Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            cache_folder=str(config.MODELS_DIR)
        )
        
        # Initialize GGUF LLM with llama.cpp
        print(f"[RAGSystem] Loading GGUF LLM: {config.LLM_MODEL}...")
        self._load_gguf_llm()
        
        # Database path (don't create connection here)
        self.db_path = str(config.SQLITE_DB_PATH)
        
        # Thread-local storage for database connections
        self._local = threading.local()
        
        # Verify collections exist (using temporary connection)
        self._verify_collections()
        
        # Initialize crisis classifier
        print("[RAGSystem] Loading crisis classifier...")
        self.crisis_classifier = get_crisis_classifier(lightweight=lightweight_classifier)
        
        # Conversation history (thread-safe with lock)
        self.conversation_history: List[Dict[str, str]] = []
        self._history_lock = threading.Lock()
        
        print("[RAGSystem] Initialization complete!\n")
    
    def _get_db_connection(self):
        """
        Get or create a thread-local database connection.
        This ensures each thread has its own SQLite connection.
        """
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            # Create new connection for this thread
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # Allow connection in different threads
            )
            self._local.cursor = self._local.conn.cursor()
        
        return self._local.conn, self._local.cursor
    
    def _load_gguf_llm(self):
        """Load GGUF quantized LLM using llama.cpp"""
        try:
            # Download GGUF model from Hugging Face if not exists
            from huggingface_hub import hf_hub_download
            
            model_path = config.MODELS_DIR / config.LLM_MODEL_FILE
            
            if not model_path.exists():
                print(f"[RAGSystem] Downloading GGUF model from {config.LLM_MODEL}...")
                model_path = hf_hub_download(
                    repo_id=config.LLM_MODEL,
                    filename=config.LLM_MODEL_FILE,
                    cache_dir=str(config.MODELS_DIR)
                )
            else:
                model_path = str(model_path)
            
            # Load with llama.cpp
            self.llm = Llama(
                model_path=model_path,
                n_ctx=config.N_CTX,           # Context window
                n_threads=config.N_THREADS,    # CPU threads
                n_gpu_layers=config.N_GPU_LAYERS,  # GPU layers (0 for CPU)
                verbose=False
            )
            
            print(f"[RAGSystem] GGUF model loaded successfully from {model_path}")
        
        except Exception as e:
            print(f"[RAGSystem] Error loading GGUF model: {e}")
            raise
    
    def _verify_collections(self):
        """Verify that required tables exist (using temporary connection)"""
        # Use temporary connection for verification
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        
        table_names = [t[0] for t in tables]
        
        if config.GENERAL_TABLE_NAME in table_names:
            count = cursor.execute(
                f"SELECT COUNT(*) FROM {config.GENERAL_TABLE_NAME}"
            ).fetchone()[0]
            print(f"[RAGSystem] General collection: {count} items")
        else:
            print("[RAGSystem] Warning: General collection not found. Run offline_indexing_optimized.py first.")
        
        if config.CRISIS_TABLE_NAME in table_names:
            count = cursor.execute(
                f"SELECT COUNT(*) FROM {config.CRISIS_TABLE_NAME}"
            ).fetchone()[0]
            print(f"[RAGSystem] Crisis collection: {count} items")
        else:
            print("[RAGSystem] Warning: Crisis collection not found.")
        
        conn.close()
    
    def deserialize_f32(self, blob: bytes) -> np.ndarray:
        """Deserialize bytes to float32 vector"""
        n = len(blob) // 4
        return np.array(struct.unpack(f'{n}f', blob))
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for user query
        
        Args:
            text: User query text
        
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding
    
    def retrieve_context(self, query: str, is_crisis: bool = False, top_k: int = None) -> List[str]:
        """
        Retrieve relevant context from sqlite-vec database
        THREAD-SAFE: Uses thread-local DB connection
        
        Args:
            query: User query
            is_crisis: Whether this is a crisis situation
            top_k: Number of results to retrieve
        
        Returns:
            List of retrieved context strings
        """
        if top_k is None:
            top_k = config.CRISIS_TOP_K if is_crisis else config.TOP_K_RETRIEVAL
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Choose table based on crisis status
        table_name = config.CRISIS_TABLE_NAME if is_crisis else config.GENERAL_TABLE_NAME
        
        print(f"[RAGSystem] Retrieving from {table_name} (top {top_k})...")
        
        # Get thread-local database connection
        conn, cursor = self._get_db_connection()
        
        # Fetch all embeddings and texts
        rows = cursor.execute(
            f"SELECT text, embedding FROM {table_name}"
        ).fetchall()
        
        if not rows:
            print("[RAGSystem] No documents found")
            return []
        
        # Calculate similarities
        similarities = []
        for text, embedding_blob in rows:
            doc_embedding = self.deserialize_f32(embedding_blob)
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((text, similarity))
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        contexts = [text for text, _ in similarities[:top_k]]
        
        print(f"[RAGSystem] Retrieved {len(contexts)} context chunks")
        return contexts
    
    def build_prompt(self, user_query: str, contexts: List[str], is_crisis: bool = False) -> str:
        """
        Build prompt for LLM with retrieved context and conversation history
        
        Args:
            user_query: Current user query
            contexts: Retrieved context chunks
            is_crisis: Whether this is a crisis situation
        
        Returns:
            Formatted prompt string
        """
        # System instructions
        if is_crisis:
            system_msg = """You are a compassionate mental health counselor specializing in crisis intervention.
Your primary goal is to provide immediate emotional support, validate feelings, and encourage professional help.
Be empathetic, non-judgmental, and focus on safety. Keep responses concise and supportive."""
        else:
            system_msg = """You are a supportive mental health counselor. Provide empathetic, helpful guidance
for mental health concerns. Be warm, understanding, and encourage healthy coping strategies.
Keep responses conversational and supportive."""
        
        # Build context section
        context_section = ""
        if contexts:
            context_section = "\n\nRelevant counseling examples:\n"
            for i, ctx in enumerate(contexts[:3], 1):  # Limit to top 3
                context_section += f"\nExample {i}:\n{ctx}\n"
        
        # Build conversation history (thread-safe access)
        history_section = ""
        with self._history_lock:
            if self.conversation_history:
                history_section = "\n\nRecent conversation:\n"
                for turn in self.conversation_history[-config.MAX_HISTORY_LENGTH:]:
                    history_section += f"User: {turn['user']}\nCounselor: {turn['assistant']}\n\n"
        
        # Construct full prompt for Gemma 2
        prompt = f"""<start_of_turn>user
{system_msg}

{context_section}

{history_section}

Current user message: {user_query}

Provide a supportive, empathetic response as a mental health counselor.<end_of_turn>
<start_of_turn>model
"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response from GGUF LLM using llama.cpp
        
        Args:
            prompt: Formatted prompt
        
        Returns:
            Generated response text
        """
        try:
            # Generate with llama.cpp
            output = self.llm(
                prompt,
                max_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                repeat_penalty=config.REPETITION_PENALTY,
                stop=["<end_of_turn>", "User:", "user"],
                echo=False
            )
            
            # Extract generated text
            response = output['choices'][0]['text'].strip()
            
            return response
        
        except Exception as e:
            print(f"[RAGSystem] Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def chat(self, user_message: str) -> Dict[str, any]:
        """
        Main chat interface - handles complete conversation flow
        THREAD-SAFE for Gradio UI
        
        Args:
            user_message: User's input message
        
        Returns:
            Dictionary with response and metadata
        """
        print(f"\n[RAGSystem] Processing: '{user_message[:50]}...'")
        
        # Step 1: Crisis Detection
        is_crisis, crisis_confidence = self.crisis_classifier.classify(user_message)
        
        # Step 2: Handle crisis situation
        if is_crisis:
            print(f"[RAGSystem] 🚨 CRISIS DETECTED (confidence: {crisis_confidence:.2f})")
            
            # Retrieve crisis-specific context
            crisis_contexts = self.retrieve_context(user_message, is_crisis=True)
            
            # Build crisis response
            crisis_response = build_crisis_response(user_message, crisis_confidence)
            
            # Also generate empathetic AI response
            prompt = self.build_prompt(user_message, crisis_contexts, is_crisis=True)
            ai_response = self.generate_response(prompt)
            
            # Combine: AI empathy + resources
            full_response = f"{ai_response}\n\n{crisis_response}"
            
            # Update history (thread-safe)
            with self._history_lock:
                self.conversation_history.append({
                    'user': user_message,
                    'assistant': full_response
                })
            
            return {
                'response': full_response,
                'is_crisis': True,
                'confidence': crisis_confidence,
                'resources_shown': True
            }
        
        # Step 3: Normal conversation path
        else:
            print("[RAGSystem] Normal conversation flow")
            
            # Retrieve context
            contexts = self.retrieve_context(user_message, is_crisis=False)
            
            # Build prompt
            prompt = self.build_prompt(user_message, contexts, is_crisis=False)
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Update history (thread-safe)
            with self._history_lock:
                self.conversation_history.append({
                    'user': user_message,
                    'assistant': response
                })
            
            return {
                'response': response,
                'is_crisis': False,
                'confidence': crisis_confidence,
                'resources_shown': False
            }
    
    def reset_conversation(self):
        """Clear conversation history (thread-safe)"""
        with self._history_lock:
            self.conversation_history = []
        print("[RAGSystem] Conversation history reset")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history (thread-safe)"""
        with self._history_lock:
            return self.conversation_history.copy()
    
    def close(self):
        """Close all database connections"""
        # Close thread-local connection if exists
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
        print("[RAGSystem] Database connections closed")

# Convenience function for quick initialization
def create_rag_system(lightweight: bool = False) -> RAGSystem:
    """
    Factory function to create optimized RAG system
    
    Args:
        lightweight: Use lightweight crisis classifier
    
    Returns:
        Initialized RAG system
    """
    return RAGSystem(lightweight_classifier=lightweight)