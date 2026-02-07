"""Visualization utilities for Hybrid RAG evaluation outputs.

Generates and saves plots for:
- metric comparisons (MRR, BERTScore, Contextual Precision, Hallucination)
- score distributions (histograms/kde)
- retrieval heatmaps (found rank buckets by question type)
- response time distributions
- ablation study charts (MRR vs config)

Usage:
    python src/visualizations.py --metrics data/metrics_per_question.json --ablation data/ablation_results.json --out data/plots --all

Plots are saved as PNG files in the output directory.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

# Project root import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# plotting libs (best-effort)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def plot_metric_comparison(records: List[Dict], out_dir: str):
    """Bar chart of average metrics across dataset."""
    meds = {}
    N = len(records)
    if N == 0:
        print('No records for metric comparison')
        return
    # metrics expected in records: 'mrr_score', 'bertscore_f1', 'contextual_precision', 'hallucination_fraction'
    sums = defaultdict(float)
    for r in records:
        sums['MRR'] += float(r.get('mrr_score', 0.0))
        sums['BERTScore'] += float(r.get('bertscore_f1', 0.0))
        sums['ContextualPrecision'] += float(r.get('contextual_precision', 0.0))
        sums['Hallucination'] += float(r.get('hallucination_fraction', 0.0))
    avgs = {k: v / N for k, v in sums.items()}

    if not MATPLOTLIB_AVAILABLE:
        print('matplotlib missing; skipping metric comparison plot')
        return
    labels = list(avgs.keys())
    values = [avgs[l] for l in labels]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
    plt.title('Average Evaluation Metrics')
    plt.ylabel('Average')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha='center', va='bottom')
    path = os.path.join(out_dir, 'metric_comparison.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print('Saved', path)


def plot_score_distributions(records: List[Dict], out_dir: str):
    """Histograms / KDE for BERTScore, Contextual Precision, Hallucination, response times."""
    if not MATPLOTLIB_AVAILABLE:
        print('matplotlib missing; skipping distributions')
        return

    def values(key):
        vals = []
        for r in records:
            v = r.get(key)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        return vals

    # BERTScore
    b = values('bertscore_f1')
    if b:
        plt.figure(figsize=(6, 4))
        if SEABORN_AVAILABLE:
            sns.histplot(b, kde=True, color='#1f77b4')
        else:
            plt.hist(b, bins=30, color='#1f77b4', alpha=0.8)
        plt.title('BERTScore F1 Distribution')
        plt.xlabel('BERTScore F1')
        plt.tight_layout()
        p = os.path.join(out_dir, 'bertscore_distribution.png')
        plt.savefig(p)
        plt.close()
        print('Saved', p)

    # Contextual Precision
    cp = values('contextual_precision')
    if cp:
        plt.figure(figsize=(6, 4))
        if SEABORN_AVAILABLE:
            sns.histplot(cp, kde=True, color='#ff7f0e')
        else:
            plt.hist(cp, bins=30, color='#ff7f0e', alpha=0.8)
        plt.title('Contextual Precision Distribution')
        plt.xlabel('Contextual Precision')
        plt.tight_layout()
        p = os.path.join(out_dir, 'contextual_precision_distribution.png')
        plt.savefig(p)
        plt.close()
        print('Saved', p)

    # Hallucination Fraction
    hf = values('hallucination_fraction')
    if hf:
        plt.figure(figsize=(6, 4))
        if SEABORN_AVAILABLE:
            sns.histplot(hf, kde=True, color='#d62728')
        else:
            plt.hist(hf, bins=30, color='#d62728', alpha=0.8)
        plt.title('Hallucination Fraction Distribution')
        plt.xlabel('Hallucination Fraction')
        plt.tight_layout()
        p = os.path.join(out_dir, 'hallucination_distribution.png')
        plt.savefig(p)
        plt.close()
        print('Saved', p)

    # Response times (may be like '0.23s' strings)
    times = []
    for r in records:
        rt = r.get('response_time') or r.get('response_ms')
        if not rt:
            continue
        if isinstance(rt, str) and rt.endswith('s'):
            try:
                times.append(float(rt[:-1]))
            except Exception:
                continue
        else:
            try:
                times.append(float(rt))
            except Exception:
                continue
    if times:
        plt.figure(figsize=(6, 4))
        if SEABORN_AVAILABLE:
            sns.histplot(times, kde=True, color='#2ca02c')
        else:
            plt.hist(times, bins=30, color='#2ca02c', alpha=0.8)
        plt.title('Response Time Distribution (s)')
        plt.xlabel('Seconds')
        plt.tight_layout()
        p = os.path.join(out_dir, 'response_time_distribution.png')
        plt.savefig(p)
        plt.close()
        print('Saved', p)


def plot_retrieval_heatmap(records: List[Dict], qa_type_map: Dict[str, str], out_dir: str, max_rank: int = 10):
    """Heatmap of counts of found_rank buckets per question type.

    found_rank should be integer or None. We bucket ranks: 1,2,3,4-5,6-10,>10,not found.
    """
    if not MATPLOTLIB_AVAILABLE:
        print('matplotlib missing; skipping heatmap')
        return

    # prepare mapping qa_type -> bucket counts
    buckets = ['1', '2', '3', '4-5', '6-10', '>10', 'not_found']
    counts_by_type = defaultdict(lambda: {b: 0 for b in buckets})
    for r in records:
        q = (r.get('question') or '').strip()
        qtype = qa_type_map.get(q, 'category')
        #qtype = r.get('qa_type') or r.get('type') or 'unknown'
        fr = r.get('found_rank') or 0
        if not fr or fr == 0:
            b = 'not_found'
        elif fr == 1:
            b = '1'
        elif fr == 2:
            b = '2'
        elif fr == 3:
            b = '3'
        elif 4 <= fr <= 5:
            b = '4-5'
        elif 6 <= fr <= 10:
            b = '6-10'
        else:
            b = '>10'
        counts_by_type[qtype][b] += 1

    types = sorted(counts_by_type.keys())
    matrix = [[counts_by_type[t][b] for b in buckets] for t in types]

    # Normalize by row (optional) to show distribution per type
    norm_matrix = []
    for row in matrix:
        s = sum(row) if sum(row) > 0 else 1
        norm_matrix.append([v / s for v in row])

    plt.figure(figsize=(max(6, len(types) * 0.6), 6))
    if SEABORN_AVAILABLE:
        sns.heatmap(norm_matrix, xticklabels=buckets, yticklabels=types, cmap='Blues', annot=True, fmt='.2f')
    else:
        plt.imshow(norm_matrix, aspect='auto', cmap='Blues')
        plt.colorbar()
        plt.xticks(range(len(buckets)), buckets, rotation=45)
        plt.yticks(range(len(types)), types)
        # annotate
        for i in range(len(types)):
            for j in range(len(buckets)):
                plt.text(j, i, f"{matrix[i][j]}", ha='center', va='center', color='black', fontsize=8)
    plt.title('Normalized retrieval rank distribution by question type')
    plt.tight_layout()
    p = os.path.join(out_dir, 'retrieval_heatmap.png')
    plt.savefig(p)
    plt.close()
    print('Saved', p)

def load_qa_types(qa_file: str) -> Dict[str, str]:
    """Return mapping question_text -> qa_type if present in QA file."""
    if not qa_file or not os.path.exists(qa_file):
        return {}
    try:
        qa_list = load_json(qa_file)
    except Exception:
        return {}
    mapping = {}
    for qa in qa_list:
        q = qa.get('question') or qa.get('Question') or ''
        qtype = qa.get('category')
        if q:
            mapping[q.strip()] = qtype
    return mapping

def plot_ablation_results(ablation_file: str, out_dir: str):
    if not os.path.exists(ablation_file):
        print('No ablation results file; skipping')
        return
    data = load_json(ablation_file).get('results') or load_json(ablation_file)
    if not data:
        print('Ablation data empty; skipping')
        return
    # Expect records with keys: k_per_method, n_final, rrf_k, dense_mrr, sparse_mrr, hybrid_mrr
    # Group by (k_per_method, n_final) and plot hybrid_mrr vs rrf_k
    groups = defaultdict(list)
    for rec in data:
        key = (rec.get('k_per_method'), rec.get('n_final'))
        groups[key].append(rec)

    if not MATPLOTLIB_AVAILABLE:
        print('matplotlib missing; skipping ablation plots')
        return

    for key, recs in groups.items():
        k, n = key
        recs_sorted = sorted(recs, key=lambda x: x.get('rrf_k', 0))
        rrfks = [r.get('rrf_k') for r in recs_sorted]
        dense = [r.get('dense_mrr') for r in recs_sorted]
        sparse = [r.get('sparse_mrr') for r in recs_sorted]
        hybrid = [r.get('hybrid_mrr') for r in recs_sorted]

        plt.figure(figsize=(6, 4))
        plt.plot(rrfks, dense, marker='o', label='dense')
        plt.plot(rrfks, sparse, marker='o', label='sparse')
        plt.plot(rrfks, hybrid, marker='o', label='hybrid')
        plt.xlabel('RRF k')
        plt.ylabel('MRR')
        plt.title(f'Ablation: k_per={k} n_final={n}')
        plt.legend()
        plt.grid(True)
        p = os.path.join(out_dir, f'ablation_k{k}_n{n}.png')
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        print('Saved', p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default='data/metrics_per_question.json')
    parser.add_argument('--qa-file', type=str, default='data/qa_generated_100.json', help='Optional QA file to map question -> qa_type')
    parser.add_argument('--ablation', type=str, default='data/ablation_results.json')
    parser.add_argument('--adversarial', type=str, default='data/adversarial_results.json')
    parser.add_argument('--out', type=str, default='data/plots')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    ensure_dir(args.out)

    # Load metrics
    records = []
    if os.path.exists(args.metrics):
        try:
            records = load_json(args.metrics)
        except Exception:
            records = []
    qa_map = load_qa_types(args.qa_file)
    if args.all or records:
        plot_metric_comparison(records, args.out)
        plot_score_distributions(records, args.out)
        plot_retrieval_heatmap(records, qa_map, args.out)

    # Ablation plots
    if args.all or os.path.exists(args.ablation):
        plot_ablation_results(args.ablation, args.out)

    # Adversarial: simple bar of hallucination_fraction by type
    if args.all and os.path.exists(args.adversarial):
        adv = load_json(args.adversarial)
        # adv is list with 'type' and 'hallucination_fraction'
        by_type = defaultdict(list)
        for a in adv:
            by_type[a.get('type', 'unknown')].append(a.get('hallucination_fraction', 0.0))
        if by_type and MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(8, 4))
            labels = list(by_type.keys())
            vals = [sum(v) / max(1, len(v)) for v in [by_type[k] for k in labels]]
            plt.bar(labels, vals, color='#9467bd')
            plt.title('Avg hallucination fraction (adversarial) by type')
            plt.ylabel('Avg hallucination fraction')
            p = os.path.join(args.out, 'adversarial_hallucination_by_type.png')
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            print('Saved', p)

    print('\nVisualizations complete. Files saved in', args.out)


if __name__ == '__main__':
    main()
