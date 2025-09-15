# Ollama Server on Kaggle with GPU Acceleration
# This script sets up an Ollama server with ngrok tunnel

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

print("Installing Ollama...")
subprocess.run("curl -fsSL https://ollama.ai/install.sh | sudo sh", shell=True)

print("Installing system dependencies...")
subprocess.run("sudo apt-get update -y", shell=True)
subprocess.run("sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers ocl-icd-opencl-dev nvidia-cuda-toolkit", shell=True)

print("Installing Python packages...")
subprocess.run(f"{sys.executable} -m pip install -q pyngrok==6.1.0 aiohttp nest_asyncio requests", shell=True)

# Verify GPU setup
print("Verifying GPU setup...")
subprocess.run("nvidia-smi", shell=True)
subprocess.run("ls /usr/local/cuda", shell=True)

# Now import the packages after installation
print("Starting main script...")
import time
import asyncio
import nest_asyncio
import aiohttp
import requests
from pyngrok import ngrok, conf

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# Configure GPU environment - ESSENTIAL FOR GPU ACCELERATION
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib64-nvidia:/usr/local/nvidia/lib64'
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['OLLAMA_GPU_LAYERS'] = '100'  # Enable GPU acceleration
os.environ["OLLAMA_SCHED_SPREAD"] = "0,1"
# Ngrok configuration - check if already defined in notebook
if 'STATIC_DOMAIN' not in globals():
    STATIC_DOMAIN = "You need to set up a free static domain from ngrok"
    print("Warning: Using default STATIC_DOMAIN. Define it in a previous cell to use your own.")
else:
    print(f"Using STATIC_DOMAIN: {STATIC_DOMAIN}")

if 'NGROK_TOKEN' not in globals():
    NGROK_TOKEN = 'This is your main auth token from ngrok'
    print("Warning: Using default NGROK_TOKEN. Define it in a previous cell to use your own.")
else:
    print("Using NGROK_TOKEN: [hidden for security]")

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
    """Create ngrok config file for static domain"""
    # Create config directory
    os.makedirs('/root/.config/ngrok', exist_ok=True)
    
    # Create config content
    config_content = f"""authtoken: {NGROK_TOKEN}
version: '2'
tunnels:
  ollama:
    proto: http
    addr: 11434
    domain: {STATIC_DOMAIN}
    host_header: rewrite
"""
    # Write config file
    with open('/root/.config/ngrok/ngrok.yml', 'w') as f:
        f.write(config_content)
    print("ngrok config file created for static domain")

async def start_ngrok():
    """Start ngrok with config file"""
    command = "ngrok start --all"
    print(f"Starting ngrok: {command}")
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Print output in real-time
    async def print_output(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            print(f"{prefix}: {line.decode().strip()}")
    
    # Monitor output
    asyncio.create_task(print_output(process.stdout, "ngrok stdout"))
    asyncio.create_task(print_output(process.stderr, "ngrok stderr"))
    
    return process

async def get_tunnel_url(timeout=60):
    """Get the tunnel URL from ngrok API"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
            if response.status_code == 200:
                tunnels = response.json()['tunnels']
                for t in tunnels:
                    if STATIC_DOMAIN in t['public_url']:
                        print(f"Tunnel verified: {t['public_url']}")
                        return t['public_url']
        except:
            pass
        await asyncio.sleep(2)
    
    print(f"Using static domain URL: https://{STATIC_DOMAIN}")
    return f"https://{STATIC_DOMAIN}"

async def main():
    # Clean up existing processes
    subprocess.run(["pkill", "-f", "ollama"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "ngrok"], stderr=subprocess.DEVNULL)
    subprocess.run(["fuser", "-k", "11434/tcp"], stderr=subprocess.DEVNULL)
    print("Cleaned up existing processes")

    # Set up ngrok config for static domain
    setup_ngrok_config()
    
    # Start Ollama server with GPU support
    print("Starting Ollama server with GPU acceleration...")
    serve_task = asyncio.create_task(run_process(['ollama', 'serve']))
    
    # Wait for Ollama API
    await wait_for_url('http://127.0.0.1:11434/v1/models')
    
    # Pull models (GPU-optimized)
    print("Pulling GPU-optimized models...")
    await run_process(['ollama', 'pull', 'deepseek-r1:14b'])
    await run_process(['ollama', 'pull', 'qwen3-coder:30b'])
    
    # Start ngrok with static domain
    ngrok_process = await start_ngrok()
    
    # Get tunnel URL
    public_url = await get_tunnel_url()
    print(f'ngrok URL: {public_url}')   
    print(f"Ready! Use this endpoint: {public_url}/api/chat")
    print("Keep this notebook running to maintain the connection")
    print("GPU acceleration is enabled")

    # Keep both processes running
    await asyncio.gather(
        serve_task,
        ngrok_process.wait()
    )

if __name__ == '__main__':
    asyncio.run(main())
