"""Compute MRR@URL-level for generated QA pairs using RRF fusion retrieval.

Usage:
  python scripts/compute_mrr.py --qa-file data/qa_generated_4.json --k-per 50 --n-final 100

This script loads QA pairs (list of dicts with `question` and `source_url`) and for each
question runs the RRF fusion retriever to obtain a ranked list of candidate chunks, then
computes the rank of the first result whose `url` matches the `source_url` (URL-level).
MRR is the average of reciprocal ranks (0 if not found).
"""
import json
import argparse
import os
import sys

# Make project root importable when running from repository root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rrf_fusion import RRFFuser


def normalize_url(u: str) -> str:
    if not u:
        return ''
    return u.rstrip('/').lower()


def compute_mrr(qa_list, k_per=50, n_final=100):
    fuser = RRFFuser()
    reciprocal_sum = 0.0
    ranks = []

    for i, qa in enumerate(qa_list, 1):
        q = qa.get('question') or qa.get('Question') or ''
        print(f"Processing QA {i}: {q}")
        true_url = normalize_url(qa.get('url'))
        print(f"True URL: {true_url}")
        fused = fuser.fuse(q, k_per, n_final)

        # Build ranked URL list (preserve order, avoid duplicates)
        seen = set()
        ranked_urls = []
        for item in fused:
            url = normalize_url(item.get('url', ''))
            if url and url not in seen:
                seen.add(url)
                ranked_urls.append(url)

        # Find rank (1-based) of true_url
        rank = None
        for idx, url in enumerate(ranked_urls, 1):
            if url == true_url:
                rank = idx
                break

        if rank:
            reciprocal_sum += 1.0 / rank
            ranks.append(rank)
        else:
            ranks.append(None)

        print(f"{i:3d}. True: {true_url} | Found rank: {rank}")

    mrr = reciprocal_sum / len(qa_list) if qa_list else 0.0
    return mrr, ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-file', type=str, default='data/qa_generated_100.json')
    parser.add_argument('--k-per', type=int, default=50, help='Top-K per method to retrieve before fusion')
    parser.add_argument('--n-final', type=int, default=100, help='Number of final fused results to consider')
    args = parser.parse_args()

    if not os.path.exists(args.qa_file):
        raise FileNotFoundError(f"QA file not found: {args.qa_file}")

    with open(args.qa_file, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)

    print(f"Computing MRR for {len(qa_list)} questions using k_per={args.k_per}, n_final={args.n_final}...")
    mrr, ranks = compute_mrr(qa_list, k_per=args.k_per, n_final=args.n_final)

    print('\n=== Results ===')
    print(f"MRR@URL = {mrr:.4f}")
    found = sum(1 for r in ranks if r)
    print(f"Found true URL in results: {found}/{len(ranks)}")


if __name__ == '__main__':
    main()
