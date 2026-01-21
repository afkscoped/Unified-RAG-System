import os
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from src.finetuning.dataset_generator import DatasetGenerator
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

def generate():
    docs_dir = "data/documents"
    pdf_path = os.path.join(docs_dir, "stats unit-4 (1).pdf")
    
    print(f"Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages[:5]: # Take first 5 pages for speed
        text += page.extract_text() + "\n"
    
    # Simple chunking
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)][:5]
    
    print(f"Generating samples for {len(chunks)} chunks...")
    gen = DatasetGenerator()
    dataset = gen.generate_qa_triplets(chunks, samples_per_chunk=2)
    
    path = gen.export_dataset(dataset, "train_v1.json")
    print(f"Dataset generated at {path}")

if __name__ == "__main__":
    generate()
