#!/usr/bin/env python3
"""
Ollama Connection Test Script
============================

This script tests if Ollama is properly installed and configured for the RAG system.
Run this before using the main RAG script to ensure everything is working.

Author: Jordan Fernandes
Date: January 2025
"""

def test_ollama_installation():
    """Test if Ollama is installed and accessible."""
    print("🔧 Testing Ollama Installation...")
    
    try:
        import ollama
        print("✅ Ollama Python package is installed")
        return True
    except ImportError:
        print("❌ Ollama Python package not found")
        print("   Install with: pip install ollama")
        return False

def test_ollama_connection():
    """Test if Ollama server is running."""
    print("\n🔗 Testing Ollama Connection...")
    
    try:
        import ollama
        models = ollama.list()
        print("✅ Successfully connected to Ollama server")
        return True, models
    except Exception as e:
        print("❌ Cannot connect to Ollama server")
        print(f"   Error: {e}")
        print("   Solutions:")
        print("   - Install Ollama: https://ollama.com/download")
        print("   - Start Ollama: run 'ollama serve' or open Ollama app")
        return False, None

def test_required_models():
    """Test if required models are installed."""
    print("\n🤖 Testing Required Models...")
    
    try:
        import ollama
        models = ollama.list()
        model_names = [model['name'].split(':')[0] for model in models['models']]
        
        required_models = ['nomic-embed-text', 'llama3.2']
        missing_models = []
        
        for model in required_models:
            if model in model_names:
                print(f"✅ {model} is available")
            else:
                print(f"❌ {model} is missing")
                missing_models.append(model)
        
        if missing_models:
            print(f"\n📥 To install missing models:")
            for model in missing_models:
                print(f"   ollama pull {model}")
            return False
        else:
            print("✅ All required models are available")
            return True
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def test_langchain_packages():
    """Test if LangChain packages are installed."""
    print("\n📦 Testing LangChain Packages...")
    
    packages = [
        ('langchain', 'langchain'),
        ('langchain_community', 'langchain-community'),
        ('langchain_ollama', 'langchain-ollama'),
        ('chromadb', 'chromadb'),
    ]
    
    missing_packages = []
    
    for package_name, pip_name in packages:
        try:
            __import__(package_name)
            print(f"✅ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} is missing")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n📥 To install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def test_pdf_file():
    """Test if the PDF file exists."""
    print("\n📄 Testing PDF File...")
    
    import os
    pdf_path = "/Users/jordanfernandes/Desktop/Dissertation workspace/datasets_pdfs/os.pdf"
    
    if os.path.exists(pdf_path):
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        print(f"✅ PDF file found: {pdf_path}")
        print(f"   File size: {file_size:.1f} MB")
        return True
    else:
        print(f"❌ PDF file not found: {pdf_path}")
        print("   Make sure your PDF file is in the correct location")
        return False

def run_full_test():
    """Run all tests and provide a summary."""
    print("🧪 RAG System Setup Test")
    print("=" * 50)
    
    tests = [
        ("Ollama Installation", test_ollama_installation),
        ("Ollama Connection", lambda: test_ollama_connection()[0]),
        ("Required Models", test_required_models),
        ("LangChain Packages", test_langchain_packages),
        ("PDF File", test_pdf_file)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your RAG system should work correctly.")
        print("   You can now run: python scripts/rag_system.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above before running the RAG system.")
        print("   Refer to OLLAMA_SETUP_GUIDE.md for detailed setup instructions.")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_test()
    exit(0 if success else 1)
