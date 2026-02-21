"""
Offline Indexing Pipeline - OPTIMIZED with sqlite-vec
Processes mental health counseling dataset and builds sqlite-vec vector database
Replaces ChromaDB with lightweight sqlite-vec for mobile deployment
"""

import sqlite3
import struct
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import config as config

class DocumentSplitter:
    """Split documents into chunks with overlap"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to split
        
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            
            # Limit chunks per document
            if len(chunks) >= config.MAX_CHUNKS_PER_DOC:
                break
        
        return chunks

class SQLiteVecIndexer:
    """
    Handles offline indexing of documents into sqlite-vec
    Lightweight alternative to ChromaDB for mobile deployment
    """
    
    def __init__(self):
        print("[SQLiteVecIndexer] Initializing...")
        
        # Initialize embedding model
        print(f"[SQLiteVecIndexer] Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            cache_folder=str(config.MODELS_DIR)
        )
        
        # Initialize sqlite-vec database
        self.db_path = str(config.SQLITE_DB_PATH)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Load sqlite-vec extension
        try:
            self.conn.enable_load_extension(True)
            self.conn.load_extension("vec0")  # sqlite-vec extension
            print("[SQLiteVecIndexer] sqlite-vec extension loaded successfully")
        except Exception as e:
            print(f"[SQLiteVecIndexer] Warning: Could not load vec0 extension: {e}")
            print("[SQLiteVecIndexer] Continuing without native vector support...")
        
        # Document splitter
        self.splitter = DocumentSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        print("[SQLiteVecIndexer] Initialization complete")
    
    def serialize_f32(self, vector: np.ndarray) -> bytes:
        """Serialize float32 vector to bytes for sqlite-vec"""
        return struct.pack(f'{len(vector)}f', *vector)
    
    def create_tables(self):
        """Create tables for general and crisis collections"""
        
        # Drop existing tables
        self.cursor.execute(f"DROP TABLE IF EXISTS {config.GENERAL_TABLE_NAME}")
        self.cursor.execute(f"DROP TABLE IF EXISTS {config.CRISIS_TABLE_NAME}")
        
        # Create general collection table
        self.cursor.execute(f"""
            CREATE TABLE {config.GENERAL_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                doc_id TEXT,
                chunk_idx INTEGER,
                source TEXT
            )
        """)
        
        # Create crisis collection table
        self.cursor.execute(f"""
            CREATE TABLE {config.CRISIS_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                doc_id TEXT,
                chunk_idx INTEGER,
                source TEXT,
                type TEXT
            )
        """)
        
        # Create indexes for faster retrieval
        self.cursor.execute(f"""
            CREATE INDEX idx_general_doc_id ON {config.GENERAL_TABLE_NAME}(doc_id)
        """)
        self.cursor.execute(f"""
            CREATE INDEX idx_crisis_doc_id ON {config.CRISIS_TABLE_NAME}(doc_id)
        """)
        
        self.conn.commit()
        print("[SQLiteVecIndexer] Tables created successfully")
    
    def load_dataset_documents(self) -> List[Dict]:
        """
        Load mental health counseling dataset from Hugging Face
        
        Returns:
            List of document dictionaries
        """
        print(f"[SQLiteVecIndexer] Loading dataset: {config.DATASET_NAME}")
        
        try:
            dataset = load_dataset(
                config.DATASET_NAME,
                split=config.DATASET_SPLIT,
                cache_dir=str(config.MODELS_DIR)
            )
            
            documents = []
            for idx, item in enumerate(dataset):
                # Extract conversation components
                question = item.get('questionText', item.get('Context', ''))
                answer = item.get('answerText', item.get('Response', ''))
                
                if question and answer:
                    # Combine as conversational pair
                    full_text = f"Question: {question}\n\nAnswer: {answer}"
                    documents.append({
                        'id': f"doc_{idx}",
                        'text': full_text,
                        'question': question,
                        'answer': answer,
                        'metadata': {
                            'source': 'counseling_conversations',
                            'doc_id': idx
                        }
                    })
            
            print(f"[SQLiteVecIndexer] Loaded {len(documents)} conversation pairs")
            return documents
        
        except Exception as e:
            print(f"[SQLiteVecIndexer] Error loading dataset: {e}")
            return []
    
    def create_general_collection(self, documents: List[Dict]):
        """
        Create general mental health conversation collection
        
        Args:
            documents: List of document dictionaries
        """
        print("[SQLiteVecIndexer] Creating general conversation collection...")
        
        # Process documents in batches
        batch_size = 100
        total_chunks = 0
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing general collection"):
            batch_docs = documents[i:i + batch_size]
            batch_data = []
            
            for doc in batch_docs:
                # Split document into chunks
                chunks = self.splitter.split_text(doc['text'])
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc['id']}_chunk_{chunk_idx}"
                    
                    # Generate embedding
                    embedding = self.embedding_model.encode(
                        chunk,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    # Serialize embedding
                    embedding_blob = self.serialize_f32(embedding)
                    
                    batch_data.append((
                        chunk_id,
                        chunk,
                        embedding_blob,
                        doc['id'],
                        chunk_idx,
                        doc['metadata']['source']
                    ))
                    
                    total_chunks += 1
            
            # Insert batch
            if batch_data:
                self.cursor.executemany(f"""
                    INSERT INTO {config.GENERAL_TABLE_NAME} 
                    (chunk_id, text, embedding, doc_id, chunk_idx, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, batch_data)
                self.conn.commit()
        
        print(f"[SQLiteVecIndexer] Indexed {total_chunks} chunks into general collection")
    
    def create_crisis_collection(self, documents: List[Dict]):
        """
        Create crisis-specific vetted scripts collection
        
        Args:
            documents: List of document dictionaries
        """
        print("[SQLiteVecIndexer] Creating crisis-specific collection...")
        
        # Filter crisis-relevant documents
        crisis_docs = []
        crisis_keywords = [
            'suicide', 'self-harm', 'crisis', 'emergency', 'immediate help',
            'hopeless', 'want to die', 'hurt myself', 'end it'
        ]
        
        for doc in documents:
            text_lower = doc['text'].lower()
            if any(keyword in text_lower for keyword in crisis_keywords):
                crisis_docs.append(doc)
        
        print(f"[SQLiteVecIndexer] Found {len(crisis_docs)} crisis-relevant documents")
        
        # Index crisis documents
        total_chunks = 0
        for doc in tqdm(crisis_docs, desc="Indexing crisis collection"):
            chunks = self.splitter.split_text(doc['text'])
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"crisis_{doc['id']}_chunk_{chunk_idx}"
                
                # Generate embedding
                embedding = self.embedding_model.encode(
                    chunk,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                # Serialize embedding
                embedding_blob = self.serialize_f32(embedding)
                
                # Insert
                self.cursor.execute(f"""
                    INSERT INTO {config.CRISIS_TABLE_NAME}
                    (chunk_id, text, embedding, doc_id, chunk_idx, source, type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    chunk,
                    embedding_blob,
                    doc['id'],
                    chunk_idx,
                    'crisis_vetted',
                    'crisis'
                ))
                
                total_chunks += 1
        
        self.conn.commit()
        print(f"[SQLiteVecIndexer] Indexed {total_chunks} crisis chunks")
    
    def build_knowledge_base(self):
        """
        Main pipeline to build complete knowledge base
        """
        print("=" * 80)
        print("OFFLINE INDEXING PIPELINE - BUILDING KNOWLEDGE BASE (sqlite-vec)")
        print("=" * 80)
        
        # Create tables
        self.create_tables()
        
        # Load documents
        documents = self.load_dataset_documents()
        if not documents:
            print("[Error] No documents loaded. Exiting.")
            return
        
        # Create general collection
        self.create_general_collection(documents)
        
        # Create crisis collection
        self.create_crisis_collection(documents)
        
        print("=" * 80)
        print("KNOWLEDGE BASE BUILD COMPLETE!")
        print("=" * 80)
        print(f"General collection: {config.GENERAL_TABLE_NAME}")
        print(f"Crisis collection: {config.CRISIS_TABLE_NAME}")
        print(f"Database location: {config.SQLITE_DB_PATH}")
        print(f"Database size: {config.SQLITE_DB_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        print("[SQLiteVecIndexer] Database connection closed")

def main():
    """Run offline indexing"""
    indexer = SQLiteVecIndexer()
    try:
        indexer.build_knowledge_base()
    finally:
        indexer.close()

if __name__ == "__main__":
    main()