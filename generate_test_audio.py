#!/usr/bin/env python3
"""
Generate a test audio file with spoken text using gTTS (Google Text-to-Speech)
This creates a real audio file for testing the Parakeet transcription service
"""

import os
import sys

print("Installing gTTS for test audio generation...")
os.system("pip install -q gtts")

from gtts import gTTS
import wave
import struct
import subprocess
from pathlib import Path

def create_test_audio(text="Hello, this is a test of the speech transcription service.", 
                     output_file="test_audio.mp3"):
    """Create a test audio file with spoken text"""
    print(f"Generating audio file: {output_file}")
    print(f"Text: {text}")
    
    # Generate speech
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)
    
    # Convert to WAV if needed (Parakeet prefers WAV)
    wav_file = output_file.replace('.mp3', '.wav')
    if output_file.endswith('.mp3'):
        print(f"Converting to WAV: {wav_file}")
        # Use ffmpeg to convert (should be available on Kaggle)
        subprocess.run([
            'ffmpeg', '-i', output_file, 
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono
            '-y',            # Overwrite
            wav_file
        ], capture_output=True)
        
        if Path(wav_file).exists():
            print(f"✓ Created {wav_file}")
            # Clean up MP3
            os.remove(output_file)
        else:
            print(f"✓ Created {output_file} (WAV conversion failed, use MP3)")
            return output_file
    
    return wav_file

def test_transcription_curl(audio_file, parakeet_url="http://localhost:8001"):
    """Generate a curl command to test the transcription"""
    print("\n" + "="*60)
    print("Test the Parakeet service with this curl command:")
    print("="*60)
    print(f"""
curl -X POST {parakeet_url}/transcribe \\
  -F "file=@{audio_file}" \\
  -F "include_timestamps=true"
""")
    
    print("\nFor ngrok URL:")
    print(f"""
curl -X POST https://parakeet-your-domain.ngrok-free.app/transcribe \\
  -F "file=@{audio_file}" \\
  -F "include_timestamps=true"
""")

if __name__ == "__main__":
    # Generate test audio files with different content
    test_phrases = [
        "Hello, this is a test of the speech transcription service.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing one two three, can you hear me clearly?",
        "Artificial intelligence is transforming how we interact with technology."
    ]
    
    audio_files = []
    for i, phrase in enumerate(test_phrases):
        filename = f"test_audio_{i+1}.mp3"
        wav_file = create_test_audio(phrase, filename)
        audio_files.append(wav_file)
        print(f"Generated: {wav_file}")
    
    print("\n" + "="*60)
    print("AUDIO FILES CREATED")
    print("="*60)
    print("Generated test audio files:")
    for f in audio_files:
        print(f"  - {f}")
    
    # Show example curl command
    if audio_files:
        test_transcription_curl(audio_files[0])
    
    print("\nYou can now use these files to test the Parakeet transcription service!")