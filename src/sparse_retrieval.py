"""
Part 1.2 Sparse Keyword Retrieval (BM25) - FIXED VERSION.
No np.save (avoids shape error); rebuilds fast on test data.
"""
import json
import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import pickle  # For simple save
nltk.download('punkt', quiet=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
#CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks_test.json')
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')
METADATA_FILE = os.path.join(DATA_DIR, 'sparse_metadata.json')
TOP_K = 5

class SparseRetriever:
    def __init__(self):
        """Initialize SparseRetriever state.

        Uses BM25Okapi to score candidate passages by token overlap.
        """
        self.bm25 = None
        self.chunks = None
        self.corpus_size = 0
        
    def build_index(self, chunks_file: str):
        """Tokenize → BM25 (no problematic np.save)."""
        print("🔄 Loading chunks...")
        with open(chunks_file, 'r') as f:
            self.chunks = json.load(f)
        
        print("📝 Tokenizing...")
        tokenized_corpus = [word_tokenize(chunk['text'].lower()) for chunk in self.chunks]
        self.corpus_size = len(tokenized_corpus)
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Save chunks only (rebuild BM25 fast)
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.chunks, f, indent=1)
        
        print(f"✅ BM25 ready: {self.corpus_size} chunks indexed!")
    
    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        """Top-K BM25 scores."""
        if self.bm25 is None:
            self.build_index(CHUNKS_FILE)  # Auto-build
        
        query_tokens = word_tokenize(query.lower())
        print("Query tokens:", query_tokens)
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            score = scores[idx]
            chunk = self.chunks[idx]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'][:200] + '...',
                'score': float(score),
                'full_text': chunk['text'],
                'url': chunk['url'],
                'title': chunk['title']
            })
        return results

def main():
    retriever = SparseRetriever()
    
    # Test
    test_queries = [
        "quantum physics mechanics",
        "world war II dates",
        "python programming language"
    ]
    
    print("\n🧪 BM25 Test:")
    for query in test_queries:
        results = retriever.retrieve(query)
        print(f"\nQ: '{query}'")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['score']:.3f} | {res['chunk_id'][:50]}...")
            print(f"   {res['url'].split('/wiki/')[-1][:40]}")
        print()

if __name__ == "__main__":
    main()
