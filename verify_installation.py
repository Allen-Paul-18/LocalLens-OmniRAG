#!/usr/bin/env python3
"""
Installation Verification Script
Tests all components of the Multimodal RAG System
"""

import sys
import subprocess
import importlib
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 9:
        print("✅ Python version OK (3.9+)")
        return True
    else:
        print("❌ Python 3.9 or higher required")
        return False

def check_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - Run: pip install {package_name}")
        return False

def check_python_packages():
    """Check all required Python packages"""
    print_header("Checking Python Packages")
    
    packages = [
        ('streamlit', 'streamlit'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('sentence-transformers', 'sentence_transformers'),
        ('chromadb', 'chromadb'),
        ('PyPDF2', 'PyPDF2'),
        ('python-docx', 'docx'),
        ('pytesseract', 'pytesseract'),
        ('Pillow', 'PIL'),
        ('SpeechRecognition', 'speech_recognition'),
        ('pydub', 'pydub'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
    ]
    
    results = []
    for package_name, import_name in packages:
        results.append(check_package(package_name, import_name))
    
    return all(results)

def check_system_command(command, package_name):
    """Check if a system command exists"""
    try:
        result = subprocess.run(
            [command, '--version'],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✅ {package_name}")
            return True
        else:
            print(f"❌ {package_name} - Not found")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"❌ {package_name} - Not installed")
        return False

def check_system_dependencies():
    """Check system-level dependencies"""
    print_header("Checking System Dependencies")
    
    tesseract_ok = check_system_command('tesseract', 'Tesseract OCR')
    ffmpeg_ok = check_system_command('ffmpeg', 'FFmpeg')
    
    if not tesseract_ok:
        print("\n⚠️  Tesseract installation:")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Ubuntu:  sudo apt-get install tesseract-ocr")
        print("   macOS:   brew install tesseract")
    
    if not ffmpeg_ok:
        print("\n⚠️  FFmpeg installation:")
        print("   Windows: https://ffmpeg.org/download.html")
        print("   Ubuntu:  sudo apt-get install ffmpeg")
        print("   macOS:   brew install ffmpeg")
    
    return tesseract_ok and ffmpeg_ok

def check_project_files():
    """Check if all project files exist"""
    print_header("Checking Project Files")
    
    required_files = [
        'streamlit_app.py',
        'multimodal_rag_system.py',
        'batch_ingest.py',
        'requirements_streamlit.txt',
        'run.py'
    ]
    
    results = []
    for filename in required_files:
        if Path(filename).exists():
            print(f"✅ {filename}")
            results.append(True)
        else:
            print(f"❌ {filename} - Missing!")
            results.append(False)
    
    return all(results)

def test_basic_imports():
    """Test if basic imports work"""
    print_header("Testing Core Imports")
    
    try:
        print("Testing RAG system import...")
        from multimodal_rag_system import MultimodalRAGSystem
        print("✅ RAG system can be imported")
        
        print("\nTesting Streamlit import...")
        import streamlit as st
        print("✅ Streamlit can be imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_model_cache():
    """Check if models are cached"""
    print_header("Checking Model Cache")
    
    cache_locations = [
        Path.home() / '.cache' / 'huggingface',
        Path.home() / '.cache' / 'torch',
    ]
    
    found_cache = False
    for cache_path in cache_locations:
        if cache_path.exists():
            print(f"✅ Cache found: {cache_path}")
            found_cache = True
    
    if not found_cache:
        print("⚠️  No model cache found")
        print("   Models will be downloaded on first run (~500MB)")
        print("   This is normal for first-time setup")
    
    return True

def create_test_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    directories = [
        './rag_storage',
        './temp_uploads',
        './demo_documents'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✅ {dir_path}")
    
    return True

def generate_report():
    """Generate installation report"""
    print_header("Installation Report")
    
    checks = {
        'Python Version': check_python_version(),
        'Python Packages': check_python_packages(),
        'System Dependencies': check_system_dependencies(),
        'Project Files': check_project_files(),
        'Core Imports': test_basic_imports(),
        'Model Cache': test_model_cache(),
        'Directories': create_test_directories()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    all_passed = all(checks.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou're ready to run the application!")
        print("\nNext steps:")
        print("  1. Run: python run.py")
        print("  2. OR: streamlit run streamlit_app.py")
        print("\nThe application will open in your browser automatically.\n")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running the application.")
        print("Refer to STREAMLIT_GUIDE.md for detailed instructions.\n")
    
    return all_passed

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  MULTIMODAL RAG SYSTEM - INSTALLATION VERIFICATION       ║
    ║  This script checks if all dependencies are installed    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    success = generate_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
