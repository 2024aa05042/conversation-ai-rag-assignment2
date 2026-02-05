"""
Part 1.1 Dense Vector Retrieval for Hybrid RAG.
- Loads chunks_test.json
- Embeds with all-MiniLM-L6-v2 (384-dim)
- Builds FAISS index
- Retrieves top-K chunks by cosine similarity
Usage: python src/dense_retrieval.py
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict
import ssl

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Config (matches test dataset)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')
INDEX_FILE = os.path.join(DATA_DIR, 'dense_index.faiss')
METADATA_FILE = os.path.join(DATA_DIR, 'dense_metadata.json')

MODEL_NAME = 'all-MiniLM-L6-v2'  # Assignment recommended[web:10]
TOP_K = 5

class DenseRetriever:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = None
        self.chunk_ids = None  # For mapping
        
    def build_index(self, chunks_file: str):
        """Embed chunks → FAISS Index."""
        print("🔄 Loading chunks...")
        with open(chunks_file, 'r') as f:
            self.chunks = json.load(f)
        
        texts = [chunk['text'] for chunk in self.chunks]
        print(f"📊 {len(texts)} chunks to embed")
        
        print("🤖 Embedding with", MODEL_NAME)
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = embeddings.astype('float32')
        
        # FAISS Index (InnerProduct = cosine sim since normalized)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine
        self.index.add(embeddings)
        
        # Save metadata mapping (index → chunk data)
        self.chunk_ids = list(range(len(self.chunks)))  # Simple ID
        metadata = [{'id': i, **chunk} for i, chunk in enumerate(self.chunks)]
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=1)
        
        # Save index
        faiss.write_index(self.index, INDEX_FILE)
        print(f"✅ Index built: {INDEX_FILE} ({self.index.ntotal} vectors, {dim}d)")
    
    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        """Retrieve top-K chunks for query."""
        if self.index is None:
            self.load_index()
        
        query_emb = self.model.encode([query], normalize_embeddings=True).astype('float32')
        scores, indices = self.index.search(query_emb, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[idx]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'][:200] + '...',  # Preview
                'score': float(score),  # Cosine sim 0-1
                'full_text': chunk['text'],
                'url': chunk['url'],
                'title': chunk['title']
            })
        return results
    
    def load_index(self):
        """Load saved index."""
        self.index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
            self.chunks = [item for item in metadata]

def main():
    retriever = DenseRetriever()
    
    # Build (first time)
    if not os.path.exists(INDEX_FILE):
        retriever.build_index(CHUNKS_FILE)
    
    # Test queries
    test_queries = [
        "What is quantum mechanics?",
        "When was World War II?",
        "Python programming features"
    ]
    
    print("\n🧪 Test Retrieval:")
    for query in test_queries:
        results = retriever.retrieve(query)
        print(f"\nQ: {query}")
        for i, res in enumerate(results, 1):
            print(f"{i}. Score: {res['score']:.3f} | {res['chunk_id'][:50]}...")
            print(f"   URL: {res['url']}")
        print()

if __name__ == "__main__":
    main()
