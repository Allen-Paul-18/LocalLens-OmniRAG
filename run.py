#!/usr/bin/env python3
"""
Quick Start Script for Multimodal RAG System
Run this to start the application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import torch
        import transformers
        import chromadb
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e.name}")
        print("\n📦 Please install requirements first:")
        print("   pip install -r requirements_streamlit.txt\n")
        return False

def check_system_deps():
    """Check if system dependencies are available"""
    issues = []
    
    # Check Tesseract
    try:
        subprocess.run(['tesseract', '--version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("Tesseract OCR - needed for image text extraction")
    
    # Check FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("FFmpeg - needed for audio processing")
    
    if issues:
        print("⚠️  Missing system dependencies:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n📖 See STREAMLIT_GUIDE.md for installation instructions\n")
        response = input("Continue anyway? (y/n): ")
        return response.lower() == 'y'
    
    return True

def main():
    print("="*60)
    print("🔍 MULTIMODAL RAG SYSTEM - STREAMLIT VERSION")
    print("="*60)
    print()
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"   Current version: {sys.version}")
        return
    
    print("✅ Python version OK")
    
    # Check dependencies
    print("📦 Checking Python packages...")
    if not check_dependencies():
        return
    print("✅ Python packages OK")
    
    # Check system dependencies
    print("🔧 Checking system dependencies...")
    if not check_system_deps():
        print("\n⚠️  Continuing with limited functionality...\n")
    else:
        print("✅ System dependencies OK")
    
    # Create necessary directories
    print("📁 Creating directories...")
    Path("./rag_storage").mkdir(exist_ok=True)
    Path("./temp_uploads").mkdir(exist_ok=True)
    print("✅ Directories ready")
    
    print("\n" + "="*60)
    print("🚀 Starting Streamlit application...")
    print("="*60)
    print("\n📌 The app will open in your browser automatically")
    print("📌 Default URL: http://localhost:8501")
    print("📌 Press Ctrl+C to stop the server\n")
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "streamlit_app.py",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        print("✅ Application stopped\n")

if __name__ == "__main__":
    main()
