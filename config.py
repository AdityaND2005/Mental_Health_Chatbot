"""
Configuration for Optimized Mobile RAG System
Uses GGUF quantized models and sqlite-vec for mobile deployment
UPDATED: Uses bartowski's GGUF repository (has actual quantized files)
"""

from pathlib import Path
import torch

# ============ PATHS ============
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
MODELS_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# ============ MODELS - OPTIMIZED FOR MOBILE ============
# LLM: GGUF Quantized Gemma 2 2B (Q4_K_M)
# FIXED: Using bartowski's repo which has actual GGUF quantized files
LLM_MODEL = "bartowski/gemma-2-2b-it-GGUF"  # Changed from google to bartowski
LLM_MODEL_FILE = "gemma-2-2b-it-Q4_K_M.gguf"  # Quantized to ~1.7GB

# Alternative quantization options (choose based on your device):
# LLM_MODEL_FILE = "gemma-2-2b-it-IQ3_M.gguf"    # ~1.39GB - for low-memory devices
# LLM_MODEL_FILE = "gemma-2-2b-it-Q4_K_S.gguf"   # ~1.64GB - slightly smaller
# LLM_MODEL_FILE = "gemma-2-2b-it-Q5_K_M.gguf"   # ~2.07GB - better quality
# LLM_MODEL_FILE = "gemma-2-2b-it-Q6_K.gguf"     # ~2.39GB - even better quality

# Embedding Model (keep sentence-transformers for now, can optimize with ONNX later)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB, fast
EMBEDDING_DIM = 384

# Crisis Classifier - lightweight
CRISIS_CLASSIFIER_MODEL = "distilbert-base-uncased"  # Lightweight for mobile

# ============ SQLITE-VEC DATABASE ============
SQLITE_DB_PATH = VECTOR_DB_DIR / "mental_health_vectors.db"
GENERAL_TABLE_NAME = "general_counseling"
CRISIS_TABLE_NAME = "crisis_intervention"

# ============ DATASET ============
DATASET_NAME = "Amod/mental_health_counseling_conversations"
DATASET_SPLIT = "train"

# ============ RETRIEVAL SETTINGS ============
TOP_K_RETRIEVAL = 3  # General conversation
CRISIS_TOP_K = 5     # Crisis situations need more context

# ============ TEXT PROCESSING ============
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_CHUNKS_PER_DOC = 5

# ============ LLM GENERATION (GGUF/llama.cpp) ============
MAX_CONTEXT_LENGTH = 2048  # Gemma 2 2B supports up to 8192, but 2048 is efficient
MAX_NEW_TOKENS = 256       # Mobile-friendly response length
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
N_CTX = 2048              # Context window for llama.cpp
N_THREADS = 4             # CPU threads (adjust for mobile)
N_GPU_LAYERS = 0          # Set to 0 for CPU, increase if mobile GPU available

# ============ CONVERSATION ============
MAX_HISTORY_LENGTH = 3  # Keep last 3 turns for context

# ============ DEVICE ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_QUANTIZATION = False  # We're using GGUF, not BitsAndBytes

# ============ CRISIS DETECTION ============
CRISIS_CONFIDENCE_THRESHOLD = 0.7
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self harm", "hurt myself", "cutting", "overdose",
    "no reason to live", "better off dead", "can't go on"
]

# ============ API SERVER ============
API_SERVER_HOST = "0.0.0.0"
API_SERVER_PORT = 5000

# ============ UI SERVER ============
UI_SHARE = False  # Set to True to enable public sharing link
UI_SERVER_NAME = "0.0.0.0"
UI_SERVER_PORT = 7860

# ============ ONNX OPTIMIZATION (OPTIONAL) ============
USE_ONNX_EMBEDDINGS = False  # Set True to use ONNX Runtime for 2-3x faster embeddings
ONNX_MODEL_PATH = MODELS_DIR / "embedding_model.onnx"

print(f"[Config] Using device: {DEVICE}")
print(f"[Config] LLM: {LLM_MODEL} (GGUF Quantized)")
print(f"[Config] Model file: {LLM_MODEL_FILE}")
print(f"[Config] Embedding: {EMBEDDING_MODEL}")
print(f"[Config] Vector DB: sqlite-vec at {SQLITE_DB_PATH}")