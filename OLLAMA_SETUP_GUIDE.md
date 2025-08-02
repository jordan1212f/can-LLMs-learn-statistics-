# Ollama Setup Guide for RAG System

## Quick Setup Steps

### 1. Install Ollama
Download and install Ollama from: https://ollama.com/download

**For macOS:**
- Download the .dmg file and install it
- Or use Homebrew: `brew install ollama`

**For Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Start Ollama
After installation, start Ollama:

**Option A: Using the app (macOS)**
- Open the Ollama app from Applications

**Option B: Using terminal**
```bash
ollama serve
```

### 3. Install Required Models
Once Ollama is running, install the models needed for the RAG system:

```bash
# Install the embedding model
ollama pull nomic-embed-text

# Install the LLM model  
ollama pull llama3.2
```

### 4. Verify Installation
Test that everything is working:

```bash
# Check if Ollama is running
ollama list

# Test the LLM
ollama run llama3.2 "Hello, how are you?"
```

### 5. Install Python Dependencies
Make sure you have the required Python packages:

```bash
pip install ollama langchain langchain-community langchain-ollama chromadb
```

### 6. Run the RAG System
Now you should be able to run the RAG system:

```bash
cd "/Users/jordanfernandes/Desktop/Dissertation workspace"
python scripts/rag_system.py
```

## Troubleshooting

### Common Issues:

1. **"Failed to connect to Ollama"**
   - Make sure Ollama is running (`ollama serve`)
   - Check if port 11434 is available

2. **"Model not found"**
   - Pull the required models: `ollama pull nomic-embed-text` and `ollama pull llama3.2`

3. **Import errors**
   - Install missing packages: `pip install ollama langchain langchain-community langchain-ollama`

4. **Permission issues**
   - On macOS, you might need to allow Ollama in System Preferences > Security & Privacy

### Check Ollama Status:
```bash
# See what models are installed
ollama list

# Check if Ollama is running
curl http://localhost:11434/api/tags

# Test connectivity
python -c "import ollama; print(ollama.list())"
```

## Next Steps
Once Ollama is set up and running, your RAG system should work properly. The script will:

1. Load and chunk your PDF documents
2. Create vector embeddings using `nomic-embed-text`
3. Set up the Q&A chain using `llama3.2`
4. Allow you to ask questions about your statistics textbooks

For any issues, check the Ollama documentation: https://github.com/ollama/ollama
