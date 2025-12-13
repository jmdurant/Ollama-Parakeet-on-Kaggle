# Ollama + Parakeet TDT Server On Kaggle

This project sets up both a GPU-accelerated Ollama LLM server and NVIDIA Parakeet TDT speech-to-text service on Kaggle with public ngrok tunnels for remote access. It is specifically designed to work within Kaggle's notebook environment with T4x2 (dual T4) GPUs.

## Features

### Integrated Services
- **Ollama LLM Server**: Large language model inference with multiple models
- **Parakeet TDT STT**: NVIDIA's high-accuracy speech-to-text transcription service
- **RAG System**: Retrieval-Augmented Generation for document-based Q&A

### Technical Features
- **Dual T4 GPU Support**: Optimized for Kaggle's T4x2 configuration with auto-distribution
- **GPU Acceleration**: Automatically configures CUDA for both services
- **Static Domain**: Uses ngrok static domains for consistent URLs
- **Multiple Models**: Pre-configured LLM models with STT capabilities
- **Non-interactive Setup**: Fully automated installation process
- **Real-time Monitoring**: Live output streaming from all services
- **Memory Management**: Intelligent GPU memory allocation between services
- **Vector Database**: ChromaDB for efficient document retrieval
- **Smart Chunking**: Specialized PDF processing for clinical documents (DSM-5, ICD codes)

## Prerequisites

- Kaggle account with T4x2 (dual T4 GPU) notebook enabled
- ngrok account with auth token
- ngrok static domain (free tier available)

## Setup Instructions

### 1. Create Kaggle Notebook
Create a new Kaggle notebook with **T4 x2** GPU enabled in the accelerator settings (select GPU T4 x2 from the accelerator dropdown).

### 2. Configure Environment Variables
Set the following variables in your Kaggle notebook:

```python
STATIC_DOMAIN = "You need to set up a free static domain from ngrok"  # Your ngrok static domain
NGROK_TOKEN = 'This is your main auth token from ngrok'               # Your ngrok authentication token
```

### 3. Choose Script Version

**Option A: Ollama Only** (Original)
- Use `main.py` for Ollama LLM server only
- Lower memory requirements, can use larger models

**Option B: Integrated Services** (New)
- Use `main_integrated.py` for both Ollama and Parakeet
- Provides both LLM chat and speech-to-text capabilities
- Optimized for T4x2 GPU configuration

### System Configuration
- Non-interactive debconf settings for automated installation
- Ollama installation using the official installer
- CUDA drivers and NVIDIA toolkit setup
- Python dependencies installation (pyngrok, aiohttp, nest_asyncio, requests)

### Environment Optimization
```bash
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['OLLAMA_GPU_LAYERS'] = '100'
os.environ["OLLAMA_SCHED_SPREAD"] = "0,1"
```
Configure this to suit your needs and demands

### Model Installation

**Ollama Only (main.py)**:
- `deepseek-r1:14b`
- `qwen3-coder:30b`

**Integrated Version (main_integrated.py)**:
- `gpt-oss:20b` (GPT-4 Turbo comparable, Mixture-of-Experts)
- `nomic-embed-text` (768-dim embeddings for RAG)
- `nvidia/parakeet-tdt-0.6b-v3` (auto-downloaded STT, latest version)
- Additional models can be pulled on-demand

## Usage

### Ollama Only Version
Access the Ollama API at:
```
https://[your-static-domain].ngrok-free.app/api/chat
```

### Integrated Version
All services available through single domain:
```
Ollama LLM: https://[your-static-domain].ngrok-free.app/api/chat
Parakeet STT: https://[your-static-domain].ngrok-free.app/transcribe
RAG Query: https://[your-static-domain].ngrok-free.app/api/rag/query
RAG Chat: https://[your-static-domain].ngrok-free.app/api/rag/chat
Document Ingest: https://[your-static-domain].ngrok-free.app/api/rag/ingest
```

The servers will remain active as long as the notebook is running.

## Testing the Services

### Automated Testing
Three test scripts are provided to verify your services are working:

1. **test_services.py** - Automated test suite for Ollama and Parakeet
```python
# Update the URLs in the script with your ngrok endpoints
python test_services.py
```

2. **test_rag.py** - Test suite for RAG functionality
```python
# Tests document ingestion, search, and RAG-enhanced chat
python test_rag.py
```

3. **generate_test_audio.py** - Creates test audio files for Parakeet
```python
python generate_test_audio.py
# This creates test WAV files you can use to test transcription
```

## API Examples

### Ollama Chat Completion
```bash
curl -X POST https://your-domain.ngrok-free.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### Ollama Generate (for telehealth-transcription-pipeline)
```bash
curl -X POST https://your-domain.ngrok-free.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "prompt": "Summarize this transcript...",
    "stream": false
  }'
```

### Parakeet Speech-to-Text
```bash
# Transcribe audio file
curl -X POST https://your-domain.ngrok-free.app/transcribe \
  -F "file=@audio.wav" \
  -F "include_timestamps=true"

# Response includes transcription and optional timestamps
```

### RAG (Retrieval-Augmented Generation)

#### Ingest Documents
```bash
# Ingest a PDF document
curl -X POST https://your-domain.ngrok-free.app/api/rag/ingest \
  -F "file=@dsm5.pdf"

# Or ingest text content directly
curl -X POST https://your-domain.ngrok-free.app/api/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "F90.0 ADHD diagnostic criteria...",
    "metadata": {"source": "DSM-5", "type": "diagnostic"}
  }'
```

#### RAG-Enhanced Chat
```bash
# Ask questions with document context
curl -X POST https://your-domain.ngrok-free.app/api/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the diagnostic criteria for ADHD?",
    "model": "gpt-oss:20b"
  }'
```

#### Search Documents
```bash
# Search for relevant chunks
curl -X POST https://your-domain.ngrok-free.app/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "attention deficit symptoms",
    "k": 5
  }'
```

#### Check RAG Status
```bash
curl https://your-domain.ngrok-free.app/api/rag/status
```

### Model Listing
```bash
curl https://your-domain.ngrok-free.app/api/tags
```

### Common Issues

1. **GPU not detected**: Verify Kaggle notebook has T4x2 GPU enabled
2. **Ngrok connection failures**: Check your auth token and static domain configuration
3. **Session timeouts**: Kaggle sessions have time limits
4. **Out of memory**: Integrated version uses smaller models to fit both services; adjust model sizes if needed
5. **Parakeet model download**: First run will download the model from HuggingFace (may take a few minutes)

### Logs and Monitoring
- The script outputs real-time logs from all services (Ollama, Parakeet, ngrok)
- GPU memory usage displayed for both T4 GPUs
- Check ngrok dashboard for tunnel status: https://dashboard.ngrok.com/

#### Adding and Removing Models

The setup script includes predefined model configurations that can be easily customized based on your requirements. To modify the models being installed, edit the model pull commands in the script:

```python
# Example model installation commands
await run_process(['ollama', 'pull', 'deepseek-r1:14b'])
await run_process(['ollama', 'pull', 'qwen3-coder:30b'])
```

**To customize your model selection:**

1. **Add models**: Insert additional `await run_process(['ollama', 'pull', 'model-name:tag'])` lines with your desired models
2. **Remove models**: Comment out or delete the model pull commands for models you don't need
3. **Replace models**: Substitute the existing model names with your preferred alternatives


Ensure any models you add are compatible with Ollama and suitable for your hardware capabilities.
Either add or remove each model name of your choice from the script. 

## GPU Memory Allocation

With T4x2 (2 × 16GB = 32GB total):
- **GPT-OSS 20B**: ~12-14GB VRAM (primary LLM)
- **Nomic-Embed-Text**: ~1-2GB VRAM (embeddings for RAG)
- **Parakeet TDT 0.6B**: ~2-3GB VRAM (speech-to-text)
- **ChromaDB & Processing**: ~1-2GB VRAM
- **System Overhead**: ~2-3GB

**Current usage: ~18-24GB, leaving ~8-14GB free** for:
- Additional models on-demand (e.g., qwen3-coder:30b)
- Larger document collections
- Multiple concurrent requests
- Other services

## Performance Benchmarks

Tested on Kaggle T4x2 (dual T4 GPUs):

### LLM Generation Speed
| Model | Size | Quantization | Generation Speed | Prompt Processing |
|-------|------|--------------|------------------|-------------------|
| **Qwen3-Coder** | 30B | Q4_K_M | **42.8 tokens/sec** | 641 tokens/sec |
| **GPT-OSS** | 20B | TBD | *To be tested* | *To be tested* |
| **DeepSeek-R1** | 14B | fp16 | **20.2 tokens/sec** | 1,031 tokens/sec |

*Note: GPT-OSS is a Mixture-of-Experts model comparable to GPT-4 Turbo for code tasks*

### Response Times
- Simple queries: 2-3 seconds
- Complex generation: 5-30 seconds depending on output length
- Parakeet transcription: Near real-time for audio processing

These speeds are production-ready for most applications, providing responsive AI services through the single domain endpoint.

## Support

For issues related to:
- Ollama: https://github.com/ollama/ollama
- Parakeet/NeMo: https://github.com/NVIDIA/NeMo
- Ngrok: https://ngrok.com/docs
- Kaggle: https://www.kaggle.com/docs/notebooks

## Guidelines for Use

This setup is intended for personal, non-commercial experimentation and learning within Kaggle's notebook environment. 
Users are responsible for ensuring their usage complies with Kaggle's Terms of Use, including restrictions on content type, resource utilization, and external connectivity. 
Be mindful of session time limits, avoid processing sensitive or regulated data, and regularly review Kaggle's policies for updates. 
This implementation should not be used for competitive activities or commercial purposes.
