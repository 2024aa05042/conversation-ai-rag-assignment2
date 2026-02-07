"""Ablation studies for Hybrid RAG retrieval.

Compares three retrieval modes:
- dense-only: top-N from DenseRetriever
- sparse-only: top-N from SparseRetriever
- hybrid (RRF): combine dense+sparse top-K with adjustable RRF k and take top-N

For each configuration, computes MRR@N (URL-level) over a QA file
and writes a summary JSON to `data/ablation_results.json`.

Usage:
    python src/ablation_studies.py --qa-file data/qa_generated_100.json --ks 5,10 --ns 10,20 --rrf-ks 10,60

Notes:
- This script expects `src/dense_retrieval.py` and `src/sparse_retrieval.py` to be available
  and will build/load indexes as needed (may take time on first run).
"""
import argparse
import json
import os
import sys
from typing import List, Dict, Tuple

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dense_retrieval import DenseRetriever
from src.sparse_retrieval import SparseRetriever
from src.compute_mrr import normalize_url


def mrr_from_ranks(ranks: List[Tuple[int, bool]]) -> float:
    """Compute mean reciprocal rank from list of ranks (None represented by 0 or None).

    `ranks` is a list of per-question rank (int) or None.
    """
    if not ranks:
        return 0.0
    total = 0.0
    for r in ranks:
        if r and isinstance(r, int) and r > 0:
            total += 1.0 / r
    return total / len(ranks)


def fuse_rrf(dense_res: List[Dict], sparse_res: List[Dict], rrf_k: int, n_final: int) -> List[Dict]:
    """Fuse dense and sparse lists using RRF with adjustable `rrf_k`.

    Each input list is ordered (rank 1..). Returns top `n_final` fused items enriched
    with `rrf_score`, `dense_rank`, `sparse_rank`, `url`, `title`, and `full_text` when present.
    """
    # Build rank maps from method results: chunk_id -> rank (1-based)
    dense_ranks = {res['chunk_id']: i + 1 for i, res in enumerate(dense_res)}
    sparse_ranks = {res['chunk_id']: i + 1 for i, res in enumerate(sparse_res)}

    all_ids = set(list(dense_ranks.keys()) + list(sparse_ranks.keys()))
    rrf_scores = {}
    # RRF scoring: sum of 1/(k + rank) across methods for each candidate
    for cid in all_ids:
        score = 0.0
        if cid in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[cid])
        if cid in sparse_ranks:
            score += 1.0 / (rrf_k + sparse_ranks[cid])
        rrf_scores[cid] = score

    top_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_final]
    # We need chunk metadata (title/url/text) to return useful results: load from chunks.json
    chunks_file = os.path.join(PROJECT_ROOT, 'data', 'chunks.json')
    if os.path.exists(chunks_file):
        with open(chunks_file, 'r', encoding='utf-8') as f:
            all_chunks = {c['chunk_id']: c for c in json.load(f)}
    else:
        all_chunks = {}

    fused = []
    for cid, score in top_ids:
        chunk = all_chunks.get(cid, {})
        fused.append({
            'chunk_id': cid,
            'rrf_score': float(score),
            'dense_rank': dense_ranks.get(cid, 0),
            'sparse_rank': sparse_ranks.get(cid, 0),
            'url': chunk.get('url', ''),
            'title': chunk.get('title', ''),
            'full_text': chunk.get('text', '')
        })
    return fused


def rank_of_true_url(retrieved_urls: List[str], true_url: str) -> int:
    """Return 1-based rank of `true_url` in `retrieved_urls`, or 0 if not found."""
    true = normalize_url(true_url)
    for i, u in enumerate(retrieved_urls, 1):
        if normalize_url(u) == true:
            return i
    return 0


def run_ablation(
    qa_list: List[Dict],
    dense: DenseRetriever,
    sparse: SparseRetriever,
    ks: List[int],
    ns: List[int],
    rrf_ks: List[int],
    out_file: str,
    verbose: bool = True,
):
    results = []

    for k in ks:
        # Ensure retrievers have at least k indexed for per-method testing
        for n in ns:
            # Dense-only: retrieve top-n from dense
            dense_ranks = []
            for qa in qa_list:
                q = qa.get('question') or qa.get('Question') or ''
                true_url = qa.get('url') or qa.get('source_url') or qa.get('source') or ''
                dense_res = dense.retrieve(q, k=n)
                urls = [r.get('url', '') for r in dense_res]
                rnk = rank_of_true_url(urls, true_url)
                dense_ranks.append(rnk)
            dense_mrr = mrr_from_ranks(dense_ranks)

            # Sparse-only
            sparse_ranks = []
            for qa in qa_list:
                q = qa.get('question') or qa.get('Question') or ''
                true_url = qa.get('url') or qa.get('source_url') or qa.get('source') or ''
                sparse_res = sparse.retrieve(q, k=n)
                urls = [r.get('url', '') for r in sparse_res]
                rnk = rank_of_true_url(urls, true_url)
                sparse_ranks.append(rnk)
            sparse_mrr = mrr_from_ranks(sparse_ranks)

            # Hybrid: test for each rrf_k value
            for rrf_k in rrf_ks:
                hybrid_ranks = []
                # For hybrid, retrieve top-k from each method then fuse and take top-n
                for qa in qa_list:
                    q = qa.get('question') or qa.get('Question') or ''
                    true_url = qa.get('url') or qa.get('source_url') or qa.get('source') or ''
                    dense_res = dense.retrieve(q, k=k)
                    sparse_res = sparse.retrieve(q, k=k)
                    fused = fuse_rrf(dense_res, sparse_res, rrf_k=rrf_k, n_final=n)
                    urls = [item.get('url', '') for item in fused]
                    rnk = rank_of_true_url(urls, true_url)
                    hybrid_ranks.append(rnk)
                hybrid_mrr = mrr_from_ranks(hybrid_ranks)

                record = {
                    'k_per_method': k,
                    'n_final': n,
                    'rrf_k': rrf_k,
                    'dense_mrr': dense_mrr,
                    'sparse_mrr': sparse_mrr,
                    'hybrid_mrr': hybrid_mrr,
                    'num_questions': len(qa_list),
                }
                results.append(record)
                if verbose:
                    print(f"k={k} n={n} rrf_k={rrf_k} -> dense MRR={dense_mrr:.4f} sparse MRR={sparse_mrr:.4f} hybrid MRR={hybrid_mrr:.4f}")

    # Save
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as wf:
        json.dump({'results': results}, wf, indent=2)
    print(f"Ablation results written to: {out_file}")
    return results


def parse_list(s: str) -> List[int]:
    return [int(x) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-file', type=str, default='data/qa_generated_100.json')
    parser.add_argument('--ks', type=str, default='5,10', help='Comma-separated K values for per-method retrieval')
    parser.add_argument('--ns', type=str, default='10,20', help='Comma-separated N values for final evaluation')
    parser.add_argument('--rrf-ks', type=str, default='10,60', help='Comma-separated RRF k values to test')
    parser.add_argument('--out', type=str, default='data/ablation_results.json')
    args = parser.parse_args()

    if not os.path.exists(args.qa_file):
        raise FileNotFoundError(f"QA file not found: {args.qa_file}")

    with open(args.qa_file, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)

    ks = parse_list(args.ks)
    ns = parse_list(args.ns)
    rrf_ks = parse_list(args.rrf_ks)

    # Initialize retrievers (will build/load indexes as needed)
    dense = DenseRetriever()
    sparse = SparseRetriever()

    # Build indexes if not present (DenseRetriever.build_index and SparseRetriever.build_index
    # expect a chunks file path in the data folder)
    chunks_file = os.path.join(PROJECT_ROOT, 'data', 'chunks.json')
    # Dense: build/load
    try:
        dense.load_index()
    except Exception:
        dense.build_index(chunks_file)

    try:
        sparse.build_index(chunks_file)
    except Exception:
        # If sparse already built, load metadata path is used inside retrieve
        pass

    run_ablation(qa_list, dense, sparse, ks, ns, rrf_ks, args.out)


if __name__ == '__main__':
    main()
