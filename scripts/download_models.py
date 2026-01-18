#!/usr/bin/env python3
"""
Model Download Script

Downloads and caches required models for offline use.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_embedding_model():
    """Download the embedding model."""
    from sentence_transformers import SentenceTransformer
    
    print("[DL] Downloading embedding model: BAAI/bge-small-en-v1.5")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    # Test it works
    embedding = model.encode(["Test sentence"])
    print(f"[OK] Embedding model ready (dim={len(embedding[0])})")
    
    return model


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            print(f"[OK] GPU available: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")
            return True
        else:
            print("[!] No GPU detected, will use CPU")
            return False
    except ImportError:
        print("[!] PyTorch not installed")
        return False


def check_groq():
    """Check Groq API key."""
    import os
    
    if os.getenv("GROQ_API_KEY"):
        print("[OK] GROQ_API_KEY is set")
        return True
    else:
        print("[!] GROQ_API_KEY not set - set it in .env file")
        return False


def main():
    """Run all setup checks and downloads."""
    print("=" * 50)
    print("[*] Advanced RAG System - Setup Script")
    print("=" * 50 + "\n")
    
    # Check GPU
    print("1. Checking GPU...")
    check_gpu()
    print()
    
    # Check API key
    print("2. Checking Groq API key...")
    check_groq()
    print()
    
    # Download models
    print("3. Downloading models...")
    try:
        download_embedding_model()
    except Exception as e:
        print(f"[X] Failed to download model: {e}")
        sys.exit(1)
    print()
    
    # Create directories
    print("4. Creating directories...")
    dirs = [
        "data/documents",
        "data/embeddings", 
        "data/cache"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {d}")
    print()
    
    print("=" * 50)
    print("[OK] Setup complete!")
    print()
    print("Next steps:")
    print("  1. Copy .env.example to .env and add your GROQ_API_KEY")
    print("  2. Run: streamlit run src/api/streamlit_app.py")
    print("  3. Or run: uvicorn src.api.fastapi_app:app --reload")
    print("=" * 50)


if __name__ == "__main__":
    main()

