#!/usr/bin/env python3
"""
Test script for Ollama + Parakeet integrated services
Run this after the services are up to verify both are working correctly
"""

import asyncio
import aiohttp
import json
import sys
import time
import io
import numpy as np
from pathlib import Path

# Configuration - update these with your actual endpoints
OLLAMA_URL = "http://localhost:11434"  # Local testing
PARAKEET_URL = "http://localhost:8001"  # Local testing

# For ngrok URLs, uncomment and update:
# OLLAMA_URL = "https://your-domain.ngrok-free.app"
# PARAKEET_URL = "https://parakeet-your-domain.ngrok-free.app"

async def test_ollama():
    """Test Ollama LLM service"""
    print("\n" + "="*60)
    print("Testing Ollama LLM Service...")
    print("="*60)
    
    try:
        # Test 1: Check if service is alive
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_URL}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("✓ Ollama service is running")
                    print(f"  Available models: {[m['name'] for m in data.get('models', [])]}")
                else:
                    print(f"✗ Ollama service returned status {resp.status}")
                    return False
            
            # Test 2: Test chat completion
            print("\nTesting chat completion...")
            chat_data = {
                "model": "deepseek-r1:14b",
                "messages": [
                    {"role": "user", "content": "Say 'Hello, Ollama is working!' in exactly those words."}
                ],
                "stream": False
            }
            
            async with session.post(
                f"{OLLAMA_URL}/api/chat",
                json=chat_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    response_text = result.get('message', {}).get('content', '')
                    print(f"✓ Chat completion successful")
                    print(f"  Response: {response_text[:100]}...")
                    return True
                else:
                    print(f"✗ Chat completion failed with status {resp.status}")
                    return False
                    
    except aiohttp.ClientConnectionError:
        print("✗ Could not connect to Ollama service")
        print(f"  Make sure the service is running at {OLLAMA_URL}")
        return False
    except Exception as e:
        print(f"✗ Ollama test failed: {e}")
        return False

async def test_parakeet():
    """Test Parakeet STT service"""
    print("\n" + "="*60)
    print("Testing Parakeet STT Service...")
    print("="*60)
    
    try:
        # Test 1: Check health endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PARAKEET_URL}/healthz") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✓ Parakeet service is running: {data}")
                else:
                    print(f"✗ Parakeet health check returned status {resp.status}")
                    return False
            
            # Test 2: Test transcription with synthetic audio
            print("\nTesting transcription with synthetic audio...")
            
            # Create a simple synthetic WAV file in memory
            # This creates a very short silent audio file just to test the endpoint
            import wave
            import struct
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                # Set parameters: 1 channel, 2 bytes per sample, 16000 Hz
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                
                # Generate 1 second of silence (you could add actual audio here)
                duration = 1  # seconds
                samples = [0] * (16000 * duration)
                
                # Write samples
                for sample in samples:
                    wav_file.writeframes(struct.pack('<h', sample))
            
            wav_buffer.seek(0)
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('file',
                          wav_buffer,
                          filename='test.wav',
                          content_type='audio/wav')
            data.add_field('include_timestamps', 'false')
            
            async with session.post(
                f"{PARAKEET_URL}/transcribe",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print("✓ Transcription endpoint working")
                    print(f"  Response: {result}")
                    return True
                else:
                    error_text = await resp.text()
                    print(f"✗ Transcription failed with status {resp.status}")
                    print(f"  Error: {error_text}")
                    return False
                    
    except aiohttp.ClientConnectionError:
        print("✗ Could not connect to Parakeet service")
        print(f"  Make sure the service is running at {PARAKEET_URL}")
        return False
    except Exception as e:
        print(f"✗ Parakeet test failed: {e}")
        return False

async def test_websocket():
    """Test Parakeet WebSocket endpoint"""
    print("\n" + "="*60)
    print("Testing Parakeet WebSocket...")
    print("="*60)
    
    # Convert HTTP URL to WebSocket URL
    ws_url = PARAKEET_URL.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, timeout=5) as ws:
                print(f"✓ WebSocket connected to {ws_url}")
                
                # Send a test message (normally would be audio data)
                test_data = b'\x00' * 1000  # Some dummy data
                await ws.send_bytes(test_data)
                
                # Try to receive a response (with timeout)
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=2)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        print(f"✓ WebSocket response received: {msg.data}")
                    else:
                        print(f"✓ WebSocket connection working (received {msg.type})")
                except asyncio.TimeoutError:
                    print("✓ WebSocket connection established (no immediate response expected for test data)")
                
                await ws.close()
                return True
                
    except aiohttp.ClientConnectionError:
        print(f"✗ Could not connect to WebSocket at {ws_url}")
        return False
    except Exception as e:
        print(f"⚠ WebSocket test skipped or failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("INTEGRATED SERVICES TEST SUITE")
    print("="*60)
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Parakeet URL: {PARAKEET_URL}")
    
    results = []
    
    # Test Ollama
    ollama_ok = await test_ollama()
    results.append(("Ollama LLM", ollama_ok))
    
    # Test Parakeet REST
    parakeet_ok = await test_parakeet()
    results.append(("Parakeet STT", parakeet_ok))
    
    # Test Parakeet WebSocket
    ws_ok = await test_websocket()
    results.append(("Parakeet WebSocket", ws_ok))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for service, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{service:20} {status}")
    
    all_passed = all(ok for _, ok in results)
    
    if all_passed:
        print("\n🎉 All services are working correctly!")
    else:
        print("\n⚠️  Some services failed. Check the logs above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)