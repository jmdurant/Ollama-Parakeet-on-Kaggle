# Ollama + Parakeet TDT Server On Kaggle

This project sets up both a GPU-accelerated Ollama LLM server and NVIDIA Parakeet TDT speech-to-text service on Kaggle with public ngrok tunnels for remote access. It is specifically designed to work within Kaggle's notebook environment with T4x2 (dual T4) GPUs.

## Features

### Integrated Services
- **Ollama LLM Server**: Large language model inference with multiple models
- **Parakeet TDT STT**: NVIDIA's high-accuracy speech-to-text transcription service

### Technical Features
- **Dual T4 GPU Support**: Optimized for Kaggle's T4x2 configuration with auto-distribution
- **GPU Acceleration**: Automatically configures CUDA for both services
- **Static Domain**: Uses ngrok static domains for consistent URLs
- **Multiple Models**: Pre-configured LLM models with STT capabilities
- **Non-interactive Setup**: Fully automated installation process
- **Real-time Monitoring**: Live output streaming from all services
- **Memory Management**: Intelligent GPU memory allocation between services

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
- `deepseek-r1:14b` (same as standalone)
- `qwen3-coder:30b` (same as standalone)
- `nvidia/parakeet-tdt-0.6b-v2` (auto-downloaded)

## Usage

### Ollama Only Version
Access the Ollama API at:
```
https://[your-static-domain].ngrok-free.app/api/chat
```

### Integrated Version
Two endpoints available:
```
Ollama LLM: https://[your-static-domain].ngrok-free.app/api/chat
Parakeet STT: https://parakeet-[your-static-domain].ngrok-free.app/transcribe
```

The servers will remain active as long as the notebook is running.

## API Examples

### Ollama Chat Completion
```bash
curl -X POST https://your-domain.ngrok-free.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1:14b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### Parakeet Speech-to-Text
```bash
# Transcribe audio file
curl -X POST https://parakeet-your-domain.ngrok-free.app/transcribe \
  -F "file=@audio.wav" \
  -F "include_timestamps=true"

# Response includes transcription and optional timestamps
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
- **Parakeet TDT 0.6B**: ~2-3GB VRAM
- **DeepSeek-R1 14B**: ~8-10GB VRAM (fp16)
- **Qwen3-Coder 30B**: ~16-18GB VRAM (fp16)
- **Overhead & Processing**: ~2-3GB

Total usage: ~28-30GB, efficiently utilizing available VRAM while leaving some headroom.

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
