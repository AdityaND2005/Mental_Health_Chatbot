# Installation & Deployment Guide

## Complete Setup Instructions

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 5GB free disk space
- Internet connection (for initial model downloads)

### Step 1: Clone/Download Project

```bash
cd mental_health_rag
# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# This will install:
# - PyTorch (deep learning framework)
# - Transformers (Hugging Face models)
# - Sentence-Transformers (embedding model)
# - ChromaDB (vector database)
# - Flask (API server)
# - Gradio (web UI)
# - Other utilities
```
### Step 3: Huggingface login

```bash
hf auth login

# Paste your hf token
```

### Step 4: Build Knowledge Base (First Time Only)

```bash
# Index the mental health counseling dataset
python driver.py --mode index

# Or use offline_indexing.py directly
python offline_indexing.py

# This process:
# - Downloads dataset from Hugging Face (~50MB)
# - Downloads models (~2.5GB total)
# - Processes ~3000+ counseling conversations
# - Generates embeddings
# - Takes 10-30 minutes depending on system
```

---

## Usage

### Option 1: Web UI (Streamlit)

```bash
# Start web interface
python driver.py --mode ui

# Opens in browser automatically
# Or visit: http://localhost:7860
```

**Features:**
- User-friendly chat interface
- Crisis resource display
- Conversation reset button
- Real-time responses

**Best for:** End users, demos, non-technical users

---

## Advanced Options

### Lightweight Mode (Reduced Memory)

```bash
# Use rule-based crisis classifier instead of deep learning
python driver.py --mode terminal --lightweight

# Reduces memory usage by ~250MB
# Slightly less accurate crisis detection
```

### Custom Port

```bash
# API server on custom port
python driver.py --mode api --port 8080

# Web UI on custom port
python driver.py --mode ui --port 8888
```

### Testing

```bash
# Run comprehensive test suite
python test_demo.py

# Tests:
# - Crisis classifier
# - Embedding generation
# - Vector retrieval
# - LLM response generation
# - Conversation history
# - Crisis resources
```

---

## Mobile Deployment

### Android (via Termux)

```bash
# Install Termux from F-Droid
# Inside Termux:

pkg update
pkg upgrade
pkg install python python-pip
pip install -r requirements.txt

# Run normally
python driver.py --mode terminal
```

### iOS (via Pythonista or a-Shell)

1. Install Pythonista or a-Shell
2. Transfer project files
3. Install dependencies (may require workarounds)
4. Run terminal mode

### Progressive Web App (PWA)

The Gradio UI can be wrapped as a PWA:

```python
# In gradio_ui.py, modify launch():
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    enable_queue=True,
    favicon_path="icon.png"  # Add your icon
)
```

---

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: ChromaDB "Collection not found"

**Solution:**
```bash
# Rebuild knowledge base
python driver.py --mode index
```

### Issue: Out of memory errors

**Solution:**
```bash
# Use lightweight mode
python driver.py --mode terminal --lightweight

# Or reduce batch sizes in config.py
```

### Issue: Models downloading every time

**Solution:**
```bash
# Models cache in ./models/ directory
# Ensure this directory persists between runs
# Check permissions on models/ directory
```

### Issue: Slow responses

**Solution:**
- First response is slow (model loading)
- Subsequent responses faster
- Use GPU if available (auto-detected)
- Reduce `MAX_NEW_TOKENS` in config.py

---

## Configuration

### Edit `config.py` to customize:

```python
# Model selection
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/gemma-2b-it"

# RAG parameters
TOP_K_RETRIEVAL = 5  # Number of context chunks
MAX_CONTEXT_LENGTH = 2048  # Max prompt length

# LLM generation
MAX_NEW_TOKENS = 256  # Max response length
TEMPERATURE = 0.7  # Randomness (0.0 to 1.0)
TOP_P = 0.9  # Nucleus sampling

# Crisis detection
CRISIS_CONFIDENCE_THRESHOLD = 0.75  # Sensitivity

# API/UI settings
API_PORT = 5000
UI_SERVER_PORT = 7860
```

---

## Project Structure

```
mental_health_rag/
├── config.py                  # Configuration settings
├── crisis_classifier.py       # Crisis detection model
├── crisis_resources.py        # Indian crisis helplines
├── offline_indexing.py        # Dataset indexing pipeline
├── rag_system.py             # Main RAG logic
├── terminal_chat.py          # CLI interface
├── api_server.py             # REST API
├── gradio_ui.py              # Web UI
├── driver.py                 # Unified entry point
├── test_demo.py              # Test suite
├── quickstart.py             # Setup wizard
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── INSTALL.md                # This file
│
├── data/
│   ├── chroma_db/            # Vector database (generated)
│   └── crisis_vectors/       # Crisis-specific vectors
│
└── models/                   # Downloaded models (generated)
    ├── sentence-transformers/
    ├── google/
    └── distilbert/
```

---

## Model Information

### Embedding Model
- **Name:** sentence-transformers/all-MiniLM-L6-v2
- **Size:** 80MB
- **Purpose:** Convert text to vectors for semantic search
- **Speed:** Fast inference on CPU
- **Quality:** Good semantic understanding

### LLM (Language Model)
- **Name:** google/gemma-2b-it
- **Size:** 2GB (quantized to 4-bit: ~600MB)
- **Purpose:** Generate conversational responses
- **Type:** Instruction-tuned for chat
- **Context:** 8K tokens

### Crisis Classifier
- **Base:** distilbert-base-uncased
- **Size:** 250MB (66MB quantized)
- **Purpose:** Detect mental health crises
- **Method:** Hybrid (keywords + deep learning)
- **Accuracy:** High sensitivity for crisis detection

---

## Data Privacy & Security

### Privacy Features
✅ **Completely Offline:** Runs locally, no data sent to cloud  
✅ **No Telemetry:** Analytics disabled in all components  
✅ **Local Storage:** Conversations stored only on device  
✅ **No Authentication:** No user tracking or accounts  

### Security Considerations
- All data processing happens locally
- No internet connection required after setup
- Vector database encrypted at rest (optional)
- API should be behind firewall in production
- Use HTTPS for API in production (add nginx/caddy)

---

## Performance Benchmarks

### On Laptop (Intel i5, 8GB RAM, CPU only)
- **First response:** ~15-30 seconds (model loading)
- **Subsequent responses:** ~3-5 seconds
- **Memory usage:** ~2.5GB RAM
- **Disk space:** ~3GB after setup

### On Desktop (Nvidia GPU, 16GB RAM)
- **First response:** ~10 seconds
- **Subsequent responses:** ~1-2 seconds
- **Memory usage:** ~3GB RAM + 2GB VRAM
- **Speed improvement:** 3-5x faster

### On Mobile (Android via Termux)
- **First response:** ~60-120 seconds
- **Subsequent responses:** ~10-20 seconds
- **Memory usage:** ~2GB RAM
- **Note:** Lightweight mode recommended

---

## Updating the System

### Update Models
```bash
# Delete models directory
rm -rf models/

# Re-download on next run
python driver.py --mode terminal
```

### Update Dataset
```bash
# Rebuild knowledge base
python driver.py --mode index
```

### Update Code
```bash
# Pull latest changes (if using git)
git pull

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## Support & Resources

### Crisis Helplines (India)
- **KIRAN:** 1800-599-0019 (24/7)
- **Vandrevala Foundation:** 1860-266-2345 (24/7)
- **iCall - TISS:** 022-2556-3291 (Mon-Sat, 8 AM - 10 PM)
- **Emergency:** 112

### Technical Support
- Check `test_demo.py` for diagnostics
- Review logs in `rag_system.log`
- Verify models in `models/` directory
- Check ChromaDB in `data/chroma_db/`

### Dataset Source
- **Hugging Face:** [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)

---

## License & Disclaimer

⚠️ **Important Disclaimers:**

1. **Not a Substitute for Professional Help:** This chatbot is a supportive tool, NOT a replacement for licensed mental health professionals.

2. **Emergency Situations:** In crisis situations, immediately contact emergency services or crisis helplines.

3. **No Medical Advice:** The system does not provide medical diagnosis or treatment recommendations.

4. **Use at Own Risk:** The developers assume no liability for outcomes from using this software.

5. **Data Accuracy:** Responses are generated by AI and may not always be accurate or appropriate.

---

## Acknowledgments

- **Dataset:** Amod/mental_health_counseling_conversations
- **Models:** Google (Gemma), Sentence-Transformers, Hugging Face
- **Vector DB:** ChromaDB
- **Crisis Resources:** Various Indian mental health organizations
