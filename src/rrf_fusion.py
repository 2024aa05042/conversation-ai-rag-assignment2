"""
Part 1.3 RRF Fusion: Combines Dense + Sparse retrieval.
Formula: RRF(d) = 1/(k + rank(d)), k=60
Selects top-N fused chunks.
"""
import json
import os
from typing import List, Dict
import numpy as np
from dense_retrieval import DenseRetriever  # Reuse 1.1
from sparse_retrieval import SparseRetriever  # Reuse 1.2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
#CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks_test.json')
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')
RRF_K = 60  # Assignment spec[file:9]
TOP_K_PER_METHOD = 10  # Retrieve more → fuse better
TOP_N_FUSED = 5  # Final output

class RRFFuser:
    def __init__(self):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        
    def fuse(self, query: str, k_per: int = TOP_K_PER_METHOD, n_final: int = TOP_N_FUSED) -> List[Dict]:
        """Get top-K dense + sparse, fuse by RRF, return top-N."""
        # Retrieve
        dense_results = self.dense.retrieve(query, k_per)
        sparse_results = self.sparse.retrieve(query, k_per)
        
        # Extract unique chunk_ids + ranks (1-based)
        dense_ranks = {res['chunk_id']: i+1 for i, res in enumerate(dense_results)}
        sparse_ranks = {res['chunk_id']: i+1 for i, res in enumerate(sparse_results)}
        
        # All unique chunks
        all_chunk_ids = set(dense_ranks) | set(sparse_ranks)
        
        # RRF scores
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1 / (RRF_K + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                score += 1 / (RRF_K + sparse_ranks[chunk_id])
            rrf_scores[chunk_id] = score
        
        # Top-N by RRF
        top_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_final]
        
        # Enrich with full data (load chunks)
        with open(CHUNKS_FILE, 'r') as f:
            all_chunks = {c['chunk_id']: c for c in json.load(f)}
        
        results = []
        for chunk_id, rrf_score in top_chunks:
            chunk = all_chunks[chunk_id]
            # Find original ranks/scores
            dense_rank = dense_ranks.get(chunk_id, 0)
            sparse_rank = sparse_ranks.get(chunk_id, 0)
            
            results.append({
                'chunk_id': chunk_id,
                'text': chunk['text'][:200] + '...',
                'full_text': chunk['text'],           # ADD THIS LINE (full for generation)
                'rrf_score': float(rrf_score),
                'dense_rank': dense_rank,
                'sparse_rank': sparse_rank,
                'url': chunk['url'],
                'title': chunk['title']
            })
        return results

def main():
    fuser = RRFFuser()
    
    test_queries = [
        "quantum mechanics principles",
        "world war II timeline",
        "python programming syntax"
    ]
    
    print("🔗 RRF Fusion (Dense + Sparse):")
    for query in test_queries:
        fused = fuser.fuse(query)
        print(f"\nQ: {query}")
        print("Rank | RRF | Dense | Sparse | Title")
        print("-" * 60)
        for i, res in enumerate(fused, 1):
            print(f"{i:4d} | {res['rrf_score']:.4f} | {res['dense_rank']:5d} | {res['sparse_rank']:5d} | {res['chunk_id'][:50]}")
        print()

if __name__ == "__main__":
    main()
