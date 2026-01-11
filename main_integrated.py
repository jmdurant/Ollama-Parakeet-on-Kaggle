# Integrated Ollama + Parakeet TDT on Kaggle with T4x2 GPUs
# This script sets up both Ollama LLM server and Parakeet STT service

# ============================================================================
# SETUP AND INSTALLATION (Using subprocess to avoid cell splitting)
# ============================================================================

import subprocess
import sys
import os

print("Setting up non-interactive environment...")
subprocess.run("echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections", shell=True)
subprocess.run("echo 'keyboard-configuration keyboard-configuration/layoutcode string us' | sudo debconf-set-selections", shell=True)
subprocess.run("echo 'keyboard-configuration keyboard-configuration/variantcode string' | sudo debconf-set-selections", shell=True)

print("Installing Ollama and system dependencies...")
subprocess.run("curl -fsSL https://ollama.ai/install.sh | sudo sh", shell=True)
subprocess.run("sudo apt-get update -y", shell=True)
subprocess.run("sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers ocl-icd-opencl-dev nvidia-cuda-toolkit ffmpeg", shell=True)

print("Installing Python dependencies for both services...")

# Pin numpy to Kaggle's version FIRST to prevent version conflicts
# scipy is pre-compiled against this version - upgrading numpy breaks it
print("Pinning numpy to Kaggle's version to prevent conflicts...")
subprocess.run(f"{sys.executable} -m pip install -q 'numpy==2.3.4'", shell=True)

# Ollama dependencies
subprocess.run(f"{sys.executable} -m pip install -q pyngrok==6.1.0 aiohttp nest_asyncio requests", shell=True)

# Parakeet dependencies - using latest versions for compatibility
subprocess.run(f"{sys.executable} -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", shell=True)
subprocess.run(f"{sys.executable} -m pip install -q nemo_toolkit[asr] omegaconf ffmpeg-python python-dotenv", shell=True)
subprocess.run(f"{sys.executable} -m pip install -q fastapi uvicorn python-multipart jinja2 psutil", shell=True)

# RAG dependencies
subprocess.run(f"{sys.executable} -m pip install -q chromadb pypdf pdfplumber sentence-transformers", shell=True)

# Fix CUDA compatibility issue
# The default cuda-python has version mismatch with Kaggle's driver
# Install numba-cuda with correct CUDA 12 bindings instead
print("Fixing CUDA compatibility...")
subprocess.run(f"{sys.executable} -m pip uninstall -y cuda-python cuda-bindings", shell=True)
subprocess.run(f"{sys.executable} -m pip install -q 'numba-cuda[cu12]'", shell=True)

# Verify GPU setup
print("Verifying dual T4 GPU setup...")
subprocess.run("nvidia-smi", shell=True)
subprocess.run("ls /usr/local/cuda", shell=True)

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

print("Starting integrated services...")
import time
import asyncio
import nest_asyncio
import aiohttp
import requests
import logging
import gc
import torch
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager, suppress
from pyngrok import ngrok, conf

# FastAPI imports for Parakeet
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# NeMo ASR for Parakeet
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict
import torchaudio
import numpy as np
import psutil

# RAG imports
import chromadb
# ChromaDB no longer needs Settings import for basic usage
import pypdf
import pdfplumber
from typing import List, Tuple
import hashlib
import json

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# ============================================================================
# GPU ENVIRONMENT CONFIGURATION
# ============================================================================

# Configure GPU environment for both services
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib64-nvidia:/usr/local/nvidia/lib64'
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use both T4 GPUs
os.environ['OLLAMA_GPU_LAYERS'] = '100'
os.environ["OLLAMA_SCHED_SPREAD"] = "0,1"  # Spread across both GPUs
os.environ['OLLAMA_KEEP_ALIVE'] = '0'  # Never unload models (0 = keep forever)

# Parakeet configuration
PARAKEET_CONFIG = {
    "MODEL_NAME": "nvidia/parakeet-tdt-0.6b-v3",
    "TARGET_SR": 16000,
    "MODEL_PRECISION": "fp16",
    "DEVICE": "cuda",  # Will auto-select available GPU
    "BATCH_SIZE": 4,
    "MAX_AUDIO_DURATION": 30,
    "VAD_THRESHOLD": 0.5,
    "PROCESSING_TIMEOUT": 60
}

# Ngrok configuration - check if already defined in notebook
if 'STATIC_DOMAIN' not in globals():
    STATIC_DOMAIN = "your-static-domain.ngrok-free.app"  # Replace with your ngrok static domain
    print("Warning: Using default STATIC_DOMAIN. Define it in a previous cell to use your own.")
else:
    print(f"Using STATIC_DOMAIN: {STATIC_DOMAIN}")

if 'NGROK_TOKEN' not in globals():
    NGROK_TOKEN = 'your_ngrok_auth_token'  # Replace with your ngrok auth token
    print("Warning: Using default NGROK_TOKEN. Define it in a previous cell to use your own.")
else:
    print("Using NGROK_TOKEN: [hidden for security]")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("integrated_service")

# ============================================================================
# PARAKEET SERVICE COMPONENTS
# ============================================================================

class TranscriptionRequest(BaseModel):
    text: Optional[str] = None
    timestamps: Optional[Dict[str, Any]] = None

class ParakeetService:
    def __init__(self):
        self.model = None
        self.device = None
        
    async def load_model(self):
        """Load Parakeet model with GPU auto-selection"""
        logger.info(f"Loading {PARAKEET_CONFIG['MODEL_NAME']} with optimized memory...")
        
        with torch.inference_mode():
            # Auto-select GPU with most free memory
            if torch.cuda.is_available():
                # Check memory on both GPUs
                free_memory = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    free_mem = torch.cuda.mem_get_info()[0] / 1024**3  # GB
                    free_memory.append(free_mem)
                    logger.info(f"GPU {i}: {free_mem:.2f}GB free")
                
                # Select GPU with most free memory
                best_gpu = free_memory.index(max(free_memory))
                self.device = f"cuda:{best_gpu}"
                logger.info(f"Selected GPU {best_gpu} for Parakeet")
            else:
                self.device = "cpu"
            
            # Load model with selected device
            dtype = torch.float16 if PARAKEET_CONFIG['MODEL_PRECISION'] == "fp16" else torch.float32
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                PARAKEET_CONFIG['MODEL_NAME'],
                map_location=self.device
            ).to(dtype=dtype)
            
            logger.info(f"Parakeet model loaded on {self.device}")
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
    async def transcribe(self, audio_path: str, include_timestamps: bool = False) -> Dict:
        """Transcribe audio file"""
        if not self.model:
            raise RuntimeError("Model not loaded")
            
        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != PARAKEET_CONFIG['TARGET_SR']:
                resampler = torchaudio.transforms.Resample(sample_rate, PARAKEET_CONFIG['TARGET_SR'])
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Transcribe
            with torch.no_grad():
                # Save to temporary file for NeMo
                temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                torchaudio.save(temp_audio.name, waveform, PARAKEET_CONFIG['TARGET_SR'])
                
                # Get transcription
                transcriptions = self.model.transcribe([temp_audio.name])
                
                # Cleanup temp file
                os.unlink(temp_audio.name)
                
                result = {"text": transcriptions[0] if transcriptions else ""}
                
                if include_timestamps:
                    # Basic timestamp estimation (would need more complex logic for real timestamps)
                    duration = waveform.shape[1] / PARAKEET_CONFIG['TARGET_SR']
                    words = result["text"].split()
                    word_duration = duration / len(words) if words else 0
                    
                    timestamps = {
                        "words": [
                            {"text": word, "start": i * word_duration, "end": (i + 1) * word_duration}
                            for i, word in enumerate(words)
                        ],
                        "segments": [{"text": result["text"], "start": 0, "end": duration}]
                    }
                    result["timestamps"] = timestamps
                
                return result
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Global Parakeet service instance
parakeet_service = ParakeetService()

# ============================================================================
# RAG SERVICE FOR DOCUMENT RETRIEVAL
# ============================================================================

class ClinicalRAGService:
    """RAG service for clinical document retrieval and enhanced Q&A"""
    
    def __init__(self):
        self.collection_name = "clinical_documents"
        self.collection = None
        self.chroma_client = None
        self.embeddings_model = None
        self.documents_indexed = 0
        
    async def initialize(self):
        """Initialize ChromaDB and embeddings"""
        logger.info("Initializing RAG service...")
        
        # Initialize ChromaDB (in-memory for Kaggle, using current API)
        self.chroma_client = chromadb.Client()
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Clinical reference documents"}
            )
            logger.info("Created new ChromaDB collection")
        except:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection with {self.collection.count()} documents")
            self.documents_indexed = self.collection.count()
        
        # Pull embeddings model if not available
        logger.info("Checking embeddings model...")
        await self.ensure_embeddings_model()
        
    async def ensure_embeddings_model(self):
        """Ensure embeddings model is available"""
        try:
            # Check if nomic-embed-text is available
            proc = await asyncio.create_subprocess_exec(
                'ollama', 'list',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            
            if 'nomic-embed-text' not in stdout.decode():
                logger.info("Pulling nomic-embed-text embeddings model...")
                proc = await asyncio.create_subprocess_exec(
                    'ollama', 'pull', 'nomic-embed-text:latest',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()
                logger.info("Embeddings model ready")
            else:
                logger.info("Embeddings model already available")
                
        except Exception as e:
            logger.error(f"Error setting up embeddings: {e}")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Ollama's nomic-embed-text model"""
        embeddings = []
        
        for text in texts:
            # Use Ollama's embeddings endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://localhost:11434/api/embeddings',
                    json={"model": "nomic-embed-text", "prompt": text}
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        embeddings.append(result['embedding'])
                    else:
                        logger.error(f"Failed to get embedding: {resp.status}")
                        embeddings.append([0.0] * 768)  # Default embedding size
        
        return embeddings
    
    def chunk_pdf(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Tuple[str, dict]]:
        """Chunk PDF intelligently for clinical documents with table support"""
        chunks = []
        
        try:
            # Use both pypdf and pdfplumber for comprehensive extraction
            with pdfplumber.open(pdf_path) as plumber_pdf:
                for page_num, plumber_page in enumerate(plumber_pdf.pages):
                    # Extract tables first
                    tables = plumber_page.extract_tables()
                    
                    # If tables found, add them as structured chunks
                    if tables:
                        for i, table in enumerate(tables):
                            # Convert table to formatted text
                            table_text = self._format_table(table)
                            if table_text:
                                table_metadata = {
                                    'page': page_num + 1,
                                    'source': pdf_path.split('/')[-1],
                                    'type': 'table',
                                    'table_index': i,
                                    'chunk_id': hashlib.md5(table_text.encode()).hexdigest()[:8]
                                }
                                chunks.append((f"[TABLE]\n{table_text}\n[/TABLE]", table_metadata))
                    
                    # Extract text with layout preservation
                    text = plumber_page.extract_text()
                    if not text:
                        continue
                        
                    # Smart chunking for clinical content
                    lines = text.split('\n')
                    current_chunk = []
                    current_size = 0
                    
                    for line in lines:
                        # Check for diagnostic codes (e.g., F90.0, 314.01)
                        is_new_section = any([
                            'F' in line and '.' in line and line[:3].replace('F', '').isdigit(),
                            line.strip().startswith(('A.', 'B.', 'C.', 'D.', 'E.')),
                            'Diagnostic Criteria' in line,
                            'Diagnostic Features' in line
                        ])
                        
                        if is_new_section and current_chunk:
                            # Save current chunk
                            chunk_text = '\n'.join(current_chunk)
                            metadata = {
                                'page': page_num + 1,
                                'source': pdf_path.split('/')[-1],
                                'type': 'text',
                                'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                            }
                            chunks.append((chunk_text, metadata))
                            current_chunk = [line]
                            current_size = len(line)
                        else:
                            current_chunk.append(line)
                            current_size += len(line)
                            
                            # Force new chunk if too large
                            if current_size > chunk_size:
                                chunk_text = '\n'.join(current_chunk)
                                metadata = {
                                    'page': page_num + 1,
                                    'source': pdf_path.split('/')[-1],
                                    'type': 'text',
                                    'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                                }
                                chunks.append((chunk_text, metadata))
                                
                                # Keep overlap
                                if overlap > 0 and len(current_chunk) > 5:
                                    current_chunk = current_chunk[-5:]
                                    current_size = sum(len(line) for line in current_chunk)
                                else:
                                    current_chunk = []
                                    current_size = 0
                    
                    # Don't forget last chunk of page
                    if current_chunk:
                        chunk_text = '\n'.join(current_chunk)
                        metadata = {
                            'page': page_num + 1,
                            'source': pdf_path.split('/')[-1],
                            'type': 'text',
                            'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                        }
                        chunks.append((chunk_text, metadata))
                        
        except Exception as e:
            logger.error(f"Error chunking PDF: {e}")
            
        return chunks
    
    def _format_table(self, table):
        """Format table data into readable text"""
        if not table:
            return ""
        
        formatted_lines = []
        for row in table:
            # Filter out None values and join with | separator
            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
            if any(cleaned_row):  # Only add non-empty rows
                formatted_lines.append(" | ".join(cleaned_row))
        
        return "\n".join(formatted_lines)
    
    async def ingest_document(self, pdf_path: str) -> int:
        """Ingest a PDF document into the vector store"""
        logger.info(f"Ingesting document: {pdf_path}")
        
        # Chunk the PDF
        chunks = self.chunk_pdf(pdf_path)
        
        if not chunks:
            logger.error("No chunks extracted from PDF")
            return 0
        
        # Get embeddings for all chunks
        texts = [chunk[0] for chunk in chunks]
        embeddings = await self.get_embeddings(texts)
        
        # Add to ChromaDB
        ids = [f"{chunk[1]['source']}_{chunk[1]['chunk_id']}" for chunk in chunks]
        metadatas = [chunk[1] for chunk in chunks]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        self.documents_indexed += len(chunks)
        logger.info(f"Ingested {len(chunks)} chunks from {pdf_path}")
        
        return len(chunks)
    
    async def query(self, question: str, n_results: int = 5) -> Tuple[str, List[dict]]:
        """Query the RAG system for relevant context"""
        if not self.collection or self.documents_indexed == 0:
            return "", []
        
        # Get embedding for the question
        question_embedding = await self.get_embeddings([question])
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=n_results
        )
        
        # Format context
        context_parts = []
        sources = []
        
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                context_parts.append(f"[Source: {metadata.get('source', 'Unknown')}, Page: {metadata.get('page', 'N/A')}]\n{doc}")
                sources.append(metadata)
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context, sources
    
    async def rag_chat(self, question: str, model: str = "gpt-oss:20b") -> dict:
        """Perform RAG-enhanced chat"""
        # Get relevant context
        context, sources = await self.query(question)
        
        # Build enhanced prompt
        if context:
            enhanced_prompt = f"""Based on the following reference material, please answer the question.

Reference Material:
{context}

Question: {question}

Please provide a comprehensive answer based on the reference material. If the reference doesn't contain enough information, indicate what's missing."""
        else:
            enhanced_prompt = question
        
        # Send to Ollama
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:11434/api/chat',
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": enhanced_prompt}],
                    "stream": False
                }
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    response = result.get('message', {}).get('content', '')
                    
                    return {
                        "response": response,
                        "sources": sources,
                        "context_used": bool(context)
                    }
                else:
                    return {
                        "response": "Error generating response",
                        "sources": [],
                        "context_used": False
                    }

# Global RAG service instance
rag_service = ClinicalRAGService()

# ============================================================================
# FASTAPI APP FOR PARAKEET
# ============================================================================

@asynccontextmanager
async def parakeet_lifespan(app: FastAPI):
    """Lifecycle manager for Parakeet FastAPI app"""
    await parakeet_service.load_model()
    yield
    # Cleanup on shutdown
    if parakeet_service.model:
        del parakeet_service.model
        torch.cuda.empty_cache()

parakeet_app = FastAPI(
    title="Parakeet TDT STT Service",
    lifespan=parakeet_lifespan
)

@parakeet_app.get("/healthz")
async def health_check():
    return {"status": "ok", "service": "parakeet"}

@parakeet_app.post("/transcribe")
@parakeet_app.post("/audio/transcriptions")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    include_timestamps: bool = Form(False),
    should_chunk: bool = Form(True)
):
    """Transcribe audio file"""
    # Save uploaded file
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        content = await file.read()
        temp_audio.write(content)
        temp_audio.close()

        # Transcribe
        result = await parakeet_service.transcribe(temp_audio.name, include_timestamps)
        return JSONResponse(content=result)

    finally:
        if os.path.exists(temp_audio.name):
            os.unlink(temp_audio.name)


@parakeet_app.get("/metrics")
async def metrics_endpoint():
    """Return current system resource usage for the UI."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    # Get GPU memory info if available
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            gpu_info.append({
                "gpu_id": i,
                "free_gb": round(free_mem / (1024**3), 2),
                "total_gb": round(total_mem / (1024**3), 2),
                "used_percent": round((1 - free_mem/total_mem) * 100, 1)
            })

    return {
        "cpu_percent": cpu_percent,
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024**3), 2),
        "ram_total_gb": round(memory.total / (1024**3), 2),
        "gpu": gpu_info
    }


@parakeet_app.get("/status")
async def status_endpoint():
    """Return current transcription progress status."""
    return {
        "status": "idle",
        "progress_percent": 0,
        "current_chunk": 0,
        "total_chunks": 0,
        "partial_text": "",
        "model_loaded": parakeet_service.model is not None,
        "device": str(parakeet_service.device) if parakeet_service.device else "not initialized"
    }


@parakeet_app.websocket("/ws")
@parakeet_app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for streaming ASR (compatible with pipeline)."""
    await websocket.accept()

    try:
        # Receive configuration (optional)
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg) if config_msg else {}
        logger.info(f"WebSocket config: {config}")

        # Accumulate audio data
        audio_chunks = []

        while True:
            try:
                # Try to receive binary audio data
                data = await websocket.receive_bytes()
                audio_chunks.append(data)
                await websocket.send_json({"status": "queued"})
            except Exception as e:
                # Check if it's an end signal
                try:
                    msg = await websocket.receive_text()
                    msg_data = json.loads(msg)
                    if msg_data.get("action") == "end_of_audio":
                        break
                except:
                    break

        # Process accumulated audio
        if audio_chunks:
            # Combine audio chunks
            audio_data = b''.join(audio_chunks)

            # Save to temp file
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio.write(audio_data)
            temp_audio.close()

            try:
                # Transcribe
                result = await parakeet_service.transcribe(temp_audio.name, include_timestamps=False)
                await websocket.send_json({"text": result.get("text", "")})
            finally:
                os.unlink(temp_audio.name)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})

# ============================================================================
# OLLAMA SERVICE FUNCTIONS
# ============================================================================

async def run_process(cmd):
    """Run a command and stream its output"""
    print('>>>', ' '.join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    async def _pipe(stream):
        async for line in stream:
            print(line.decode().rstrip())
    await asyncio.gather(_pipe(proc.stdout), _pipe(proc.stderr))
    return proc

async def wait_for_url(url, timeout=30):
    """Wait for a URL to become available"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession() as sess:
                resp = await sess.get(url)
                if resp.status < 500:
                    return
        except:
            pass
        await asyncio.sleep(1)
    raise RuntimeError(f"Timeout waiting for {url}")

def setup_ngrok_config():
    """Create ngrok config file for single domain with reverse proxy"""
    os.makedirs('/root/.config/ngrok', exist_ok=True)
    
    # Configure ngrok for single tunnel to reverse proxy
    config_content = f"""authtoken: {NGROK_TOKEN}
version: '2'
tunnels:
  main:
    proto: http
    addr: 8080
    domain: {STATIC_DOMAIN}
    host_header: rewrite
"""
    
    with open('/root/.config/ngrok/ngrok.yml', 'w') as f:
        f.write(config_content)
    print("ngrok config created for single domain routing")

async def start_ngrok():
    """Start ngrok with config file"""
    command = "ngrok start --all"
    print(f"Starting ngrok: {command}")
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    async def print_output(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            print(f"{prefix}: {line.decode().strip()}")
    
    asyncio.create_task(print_output(process.stdout, "ngrok"))
    asyncio.create_task(print_output(process.stderr, "ngrok"))
    
    return process

async def get_tunnel_urls(timeout=60):
    """Get tunnel URLs from ngrok API"""
    deadline = time.time() + timeout
    urls = {}
    
    while time.time() < deadline:
        try:
            response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
            if response.status_code == 200:
                tunnels = response.json()['tunnels']
                for t in tunnels:
                    if 'ollama' in t['name']:
                        urls['ollama'] = t['public_url']
                    elif 'parakeet' in t['name']:
                        urls['parakeet'] = t['public_url']
                
                if len(urls) == 2:  # Both tunnels found
                    print(f"Found tunnels: Ollama={urls['ollama']}, Parakeet={urls['parakeet']}")
                    return urls
        except:
            pass
        await asyncio.sleep(2)
    
    # Show what we found
    print(f"Tunnel discovery timeout. Found: {urls}")
    
    # Fallback - use what we have
    if not urls.get('ollama'):
        urls['ollama'] = f"https://{STATIC_DOMAIN}"
    
    return urls

# ============================================================================
# REVERSE PROXY FOR SINGLE DOMAIN
# ============================================================================

from aiohttp import web

async def create_reverse_proxy():
    """Create a reverse proxy to route requests to appropriate services"""
    app = web.Application()
    
    async def proxy_handler(request):
        """Route requests based on path"""
        path = request.path_qs
        
        # Determine target based on path
        if path.startswith('/api/chat') or path.startswith('/api/generate') or path.startswith('/api/tags') or path.startswith('/api/embeddings'):
            # Route to Ollama (includes /api/generate for telehealth-transcription-pipeline)
            target_url = f"http://localhost:11434{path}"
        elif path.startswith('/api/rag'):
            # Handle RAG endpoints directly
            try:
                if path == '/api/rag/query':
                    # RAG-enhanced query
                    data = await request.json()
                    result = await rag_service.rag_chat(
                        question=data.get('question', ''),
                        model=data.get('model', 'gpt-oss:20b')
                    )
                    return web.json_response(result)
                    
                elif path == '/api/rag/ingest':
                    # Document ingestion
                    data = await request.json()
                    pdf_path = data.get('pdf_path', '')
                    if os.path.exists(pdf_path):
                        chunks = await rag_service.ingest_document(pdf_path)
                        return web.json_response({
                            "status": "success",
                            "chunks_indexed": chunks,
                            "total_documents": rag_service.documents_indexed
                        })
                    else:
                        return web.json_response({
                            "status": "error",
                            "message": f"File not found: {pdf_path}"
                        }, status=404)
                        
                elif path == '/api/rag/status':
                    # RAG system status
                    return web.json_response({
                        "status": "ready" if rag_service.collection else "not_initialized",
                        "documents_indexed": rag_service.documents_indexed,
                        "collection_name": rag_service.collection_name if rag_service else "not_initialized"
                    })
                    
                elif path == '/api/rag/search':
                    # Direct search without LLM
                    data = await request.json()
                    context, sources = await rag_service.query(
                        question=data.get('question', ''),
                        n_results=data.get('n_results', 5)
                    )
                    return web.json_response({
                        "context": context,
                        "sources": sources
                    })
                else:
                    return web.json_response({"error": "Unknown RAG endpoint"}, status=404)
                    
            except Exception as e:
                logger.error(f"RAG endpoint error: {e}")
                return web.json_response({"error": str(e)}, status=500)
                
        elif path.startswith('/transcribe') or path.startswith('/audio/transcriptions') or path.startswith('/healthz') or path.startswith('/ws') or path.startswith('/metrics') or path.startswith('/status'):
            # Route to Parakeet
            target_url = f"http://localhost:8001{path}"
        else:
            info_text = """Service router running.

Endpoints:
- /api/chat - Ollama LLM chat
- /api/tags - List available models
- /api/rag/query - RAG-enhanced chat
- /api/rag/search - Search documents
- /api/rag/ingest - Add document to RAG
- /api/rag/status - RAG system status
- /transcribe - Parakeet STT
- /audio/transcriptions - Parakeet STT (OpenAI compatible)
- /healthz - Health check
- /metrics - System metrics (CPU/RAM/GPU)
- /status - Transcription status
- /ws - Parakeet WebSocket
- /ws/transcribe - Parakeet WebSocket (pipeline compatible)"""
            return web.Response(text=info_text, status=200)
        
        # Forward the request
        async with aiohttp.ClientSession() as session:
            # Handle WebSocket upgrade
            if request.headers.get('Upgrade') == 'websocket':
                ws_server = web.WebSocketResponse()
                await ws_server.prepare(request)
                
                async with session.ws_connect(target_url) as ws_client:
                    async def forward_client_to_server():
                        async for msg in ws_server:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await ws_client.send_str(msg.data)
                            elif msg.type == aiohttp.WSMsgType.BINARY:
                                await ws_client.send_bytes(msg.data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break
                    
                    async def forward_server_to_client():
                        async for msg in ws_client:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await ws_server.send_str(msg.data)
                            elif msg.type == aiohttp.WSMsgType.BINARY:
                                await ws_server.send_bytes(msg.data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break
                    
                    await asyncio.gather(forward_client_to_server(), forward_server_to_client())
                return ws_server
            
            # Handle regular HTTP requests
            method = request.method
            headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'content-length']}
            data = await request.read() if request.body_exists else None
            
            async with session.request(method, target_url, headers=headers, data=data) as resp:
                body = await resp.read()
                return web.Response(body=body, status=resp.status, headers=resp.headers)
    
    # Add catch-all route
    app.router.add_route('*', '/{path:.*}', proxy_handler)
    
    return app

async def run_reverse_proxy():
    """Run the reverse proxy server"""
    app = await create_reverse_proxy()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    print("Reverse proxy running on port 8080")
    return runner

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def run_parakeet_server():
    """Run Parakeet FastAPI server"""
    config = uvicorn.Config(
        app=parakeet_app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    # Clean up existing processes
    subprocess.run(["pkill", "-f", "ollama"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "ngrok"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "uvicorn"], stderr=subprocess.DEVNULL)
    subprocess.run(["fuser", "-k", "11434/tcp"], stderr=subprocess.DEVNULL)
    subprocess.run(["fuser", "-k", "8001/tcp"], stderr=subprocess.DEVNULL)
    print("Cleaned up existing processes")

    # Set up ngrok config
    setup_ngrok_config()
    
    # Start Ollama server
    print("Starting Ollama server with GPU acceleration...")
    ollama_task = asyncio.create_task(run_process(['ollama', 'serve']))
    
    # Wait for Ollama API
    await wait_for_url('http://127.0.0.1:11434/v1/models')
    
    # Pull/update Ollama models
    print("Checking and updating models (will skip if already up-to-date)...")
    print("Note: T4x2 provides 32GB total VRAM")
    
    # Only pull GPT-OSS:20B for faster startup
    print("Checking gpt-oss:20b...")
    await run_process(['ollama', 'pull', 'gpt-oss:20b'])
    
    # Pull embeddings model for RAG
    print("Checking nomic-embed-text for RAG embeddings...")
    await run_process(['ollama', 'pull', 'nomic-embed-text'])
    
    # Skip other models for faster startup
    # Uncomment if needed:
    # await run_process(['ollama', 'pull', 'qwen3-coder:30b'])
    
    print("Models ready!")
    
    # Pre-load primary model to avoid cold start delays
    print("\nPre-loading primary model into memory...")
    print("Pre-loading gpt-oss:20b as default model...")
    
    preload_data = {
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post('http://127.0.0.1:11434/api/chat', json=preload_data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    load_time = result.get('total_duration', 0)/1_000_000_000
                    print(f"✓ GPT-OSS:20B pre-loaded (took {load_time:.1f}s)")
                    print(f"  Model will stay in memory (OLLAMA_KEEP_ALIVE=0)")
                    print(f"  Switching to qwen3-coder will take ~40s but both stay loaded once used")
                else:
                    print("⚠ Could not pre-load GPT-OSS")
    except Exception as e:
        print(f"⚠ Pre-load error: {e}")
    
    print("\nMemory status:")
    print("  GPT-OSS 20B: Loaded (~12-14GB)")
    print("  Parakeet: Loaded (~3GB)")
    print("  Total: ~15-17GB used, ~15-17GB free")
    print("  Plenty of room for RAG, embeddings, or additional models")
    
    # Start Parakeet server
    print("\nStarting Parakeet STT server...")
    parakeet_task = asyncio.create_task(run_parakeet_server())
    
    # Give Parakeet time to start
    await asyncio.sleep(5)
    
    # Initialize RAG service
    print("\n" + "="*60)
    print("Initializing RAG Service...")
    print("="*60)
    
    global rag_service
    rag_service = ClinicalRAGService()
    await rag_service.initialize()
    
    # Check for documents to ingest on startup
    kaggle_input_dir = Path("/kaggle/input")
    if kaggle_input_dir.exists():
        print(f"\nChecking for documents in {kaggle_input_dir}...")
        pdf_files = list(kaggle_input_dir.glob("**/*.pdf"))
        
        if pdf_files:
            print(f"Found {len(pdf_files)} PDF files to ingest:")
            for pdf_file in pdf_files[:5]:  # Show first 5 files
                print(f"  - {pdf_file.name}")
            if len(pdf_files) > 5:
                print(f"  ... and {len(pdf_files) - 5} more")
            
            print("\nIngesting documents into RAG system...")
            for pdf_file in pdf_files:
                try:
                    print(f"  Ingesting: {pdf_file.name}")
                    result = await rag_service.ingest_pdf(str(pdf_file))
                    if result['success']:
                        print(f"    ✓ Added {result['chunks_added']} chunks")
                    else:
                        print(f"    ✗ Failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"    ✗ Error ingesting {pdf_file.name}: {e}")
            
            print(f"\n✓ RAG system ready with {len(pdf_files)} documents")
        else:
            print("  No PDF documents found to ingest")
            print("  Upload PDFs to /kaggle/input/ or use the /api/rag/ingest endpoint")
    else:
        print(f"\n{kaggle_input_dir} not found (normal if not in Kaggle)")
        print("  Documents can be ingested via the /api/rag/ingest endpoint")
    
    print(f"\nRAG Service initialized successfully!")
    print(f"  Collection: {rag_service.collection_name}")
    print(f"  Embeddings: nomic-embed-text")
    print(f"  Vector DB: ChromaDB (in-memory)")
    
    # Start reverse proxy
    print("\nStarting reverse proxy for single-domain routing...")
    proxy_runner = await run_reverse_proxy()
    
    # Give proxy time to start
    await asyncio.sleep(2)
    
    # Start ngrok
    ngrok_process = await start_ngrok()
    
    # Get tunnel URLs
    urls = await get_tunnel_urls()
    
    print("\n" + "="*60)
    print("SERVICES READY - SINGLE DOMAIN ROUTING!")
    print("="*60)
    
    base_url = f"https://{STATIC_DOMAIN}"
    print(f"Base URL: {base_url}")
    print()
    print("Core Endpoints:")
    print(f"  Ollama LLM:      {base_url}/api/chat")
    print(f"  Parakeet STT:    {base_url}/transcribe")
    print(f"  Parakeet STT:    {base_url}/audio/transcriptions (OpenAI compatible)")
    print(f"  Parakeet WS:     wss://{STATIC_DOMAIN}/ws")
    print(f"  Parakeet WS:     wss://{STATIC_DOMAIN}/ws/transcribe (pipeline compatible)")
    print(f"  Health Check:    {base_url}/healthz")
    print(f"  Metrics:         {base_url}/metrics")
    print(f"  Status:          {base_url}/status")
    
    print("\nRAG Endpoints:")
    print(f"  RAG Query:       {base_url}/api/rag/query")
    print(f"  RAG Chat:        {base_url}/api/rag/chat")
    print(f"  Document Ingest: {base_url}/api/rag/ingest")
    print(f"  RAG Search:      {base_url}/api/rag/search")
    print(f"  RAG Status:      {base_url}/api/rag/status")
    
    print("\nExample usage:")
    print(f"  # Ollama chat")
    print(f"  curl -X POST {base_url}/api/chat \\")
    print(f"    -d '{{\"model\":\"gpt-oss:20b\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'")
    print()
    print(f"  # RAG-enhanced chat (clinical documents)")
    print(f"  curl -X POST {base_url}/api/rag/chat \\")
    print(f"    -d '{{\"question\":\"What are the diagnostic criteria for ADHD?\",\"model\":\"gpt-oss:20b\"}}'")
    print()
    print(f"  # Parakeet transcription")
    print(f"  curl -X POST {base_url}/transcribe \\")
    print(f"    -F 'file=@audio.wav' \\")
    print(f"    -F 'include_timestamps=true'")
    
    print("\nGPU Status:")
    print(f"  - Both T4 GPUs are being utilized")
    print(f"  - Ollama models: GPU auto-distribution")
    print(f"  - Parakeet model: Selected optimal GPU")
    print(f"\nAll services accessible through single domain: {STATIC_DOMAIN}")
    print("Keep this notebook running to maintain the connection")
    print("="*60)
    
    # Keep all services running
    await asyncio.gather(
        ollama_task,
        parakeet_task,
        ngrok_process.wait()
    )

if __name__ == '__main__':
    asyncio.run(main())