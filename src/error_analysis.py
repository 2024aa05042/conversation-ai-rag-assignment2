"""Error analysis and visualization for Hybrid RAG evaluation.

Categorizes per-question failures into:
- retrieval (true source not found in retrieved results),
- generation (short/empty answer or low BERTScore),
- context issues (low contextual precision or high hallucination fraction).

Aggregates counts by question type (if `--qa-file` with `qa_type` is provided)
and saves:
- `data/error_analysis_summary.json`
- `data/error_analysis_per_question.csv`
- PNG plots in `data/`:
    - `error_by_type.png`
    - `error_overall_pie.png`

Usage:
    python src/error_analysis.py --metrics data/metrics_per_question.json --qa-file data/qa_generated_100.json

This script uses matplotlib (and seaborn if available) for plotting.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
        # Accept multiple possible fields for QA type; some files use 'category' or 'qa_type'
        qtype = qa.get('category')
        if q:
            mapping[q.strip()] = qtype
    return mapping


def categorize_item(item: Dict, thresholds: Dict) -> List[str]:
    """Return list of failure labels for this item.

    Labels: 'retrieval', 'generation', 'context', or empty list for OK.
    """
    labels = []
    # retrieval failure: treat missing/zero `found_rank` as a retrieval miss
    # normalize access to the field (some outputs may use None/0/absent)
    found_rank = item.get('found_rank')
    if not found_rank:
        labels.append('retrieval')

    # generation failure: empty/short generated_answer OR low BERTScore
    gen = (item.get('generated_answer') or '').strip()
    bert = item.get('bertscore_f1')
    if not gen or len(gen) < thresholds['gen_min_len']:
        labels.append('generation')
    elif bert is not None and bert < thresholds['bert_low']:
        labels.append('generation')

    # context issue: low contextual precision or high hallucination
    ctx_prec = item.get('contextual_precision')
    halluc = item.get('hallucination_fraction')
    if ctx_prec is not None and ctx_prec < thresholds['ctx_prec_low']:
        labels.append('context')
    if halluc is not None and halluc > thresholds['halluc_high']:
        labels.append('context')

    # Remove duplicates
    print(f"Question: {item.get('question')}\nLabels: {list(set(labels))}\n Sorted Labels: {sorted(set(labels))}\n")
    return list(set(labels))


def aggregate_by_type(records: List[Dict], qa_type_map: Dict[str, str], thresholds: Dict):
    """Aggregate failure categories by question type."""
    per_type_counts = defaultdict(Counter)
    per_type_total = Counter()
    per_question_results = []

    for item in records:
        q = (item.get('question') or '').strip()
        # Map to known qa_type; default to 'unknown' when not provided in mapping
        qtype = qa_type_map.get(q, 'unknown')
        labels = categorize_item(item, thresholds)
        primary = labels[0] if labels else 'ok'

        per_type_counts[qtype][primary] += 1
        per_type_total[qtype] += 1

        # store per-question result
        r = {
            'question': q,
            'qa_type': qtype,
            'labels': labels,
            'primary_label': primary,
            'found_rank': item.get('found_rank'),
            'bertscore_f1': item.get('bertscore_f1'),
            'contextual_precision': item.get('contextual_precision'),
            'hallucination_fraction': item.get('hallucination_fraction'),
            'generated_answer': (item.get('generated_answer') or '')[:400]
        }
        per_question_results.append(r)

    return per_type_counts, per_type_total, per_question_results


def save_outputs(out_dir: str, per_type_counts, per_type_total, per_question_results):
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        'per_type_counts': {k: dict(v) for k, v in per_type_counts.items()},
        'per_type_total': dict(per_type_total)
    }
    with open(os.path.join(out_dir, 'error_analysis_summary.json'), 'w', encoding='utf-8') as wf:
        json.dump(summary, wf, indent=2)

    # CSV per-question
    csv_file = os.path.join(out_dir, 'error_analysis_per_question.csv')
    with open(csv_file, 'w', encoding='utf-8', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['question', 'qa_type', 'primary_label', 'labels', 'found_rank', 'bertscore_f1', 'contextual_precision', 'hallucination_fraction', 'generated_answer'])
        writer.writeheader()
        for r in per_question_results:
            row = {k: (','.join(v) if isinstance(v, list) else v) for k, v in r.items()}
            writer.writerow(row)

    print(f"Saved summary JSON and CSV to {out_dir}")
    return summary, csv_file


def plot_results(out_dir: str, per_type_counts, per_type_total):
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available: skipping plots")
        return

    # Prepare data for plotting
    types = sorted(per_type_total.keys())
    categories = ['retrieval', 'generation', 'context', 'ok']
    data = {cat: [per_type_counts[t].get(cat, 0) for t in types] for cat in categories}

    # stacked bar chart
    plt.figure(figsize=(max(6, len(types) * 1.2), 6))
    bottom = [0] * len(types)
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']
    for i, cat in enumerate(categories):
        plt.bar(types, data[cat], bottom=bottom, label=cat, color=colors[i])
        bottom = [b + v for b, v in zip(bottom, data[cat])]
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title('Error categories by question type')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'error_by_type.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

    # overall pie
    total_counts = Counter()
    for t in per_type_counts:
        total_counts.update(per_type_counts[t])
    labels = []
    sizes = []
    for cat in categories:
        v = total_counts.get(cat, 0)
        if v > 0:
            labels.append(cat)
            sizes.append(v)
    if sizes:
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Overall error category distribution')
        out_pie = os.path.join(out_dir, 'error_overall_pie.png')
        plt.savefig(out_pie)
        plt.close()
        print(f"Saved plot: {out_pie}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default='data/metrics_per_question.json', help='Per-question metrics JSON produced by compute_metrics')
    parser.add_argument('--qa-file', type=str, default='data/qa_generated_100.json', help='Optional QA file to map question -> qa_type')
    parser.add_argument('--out-dir', type=str, default='data', help='Directory to write outputs')
    parser.add_argument('--bert-low', type=float, default=0.15, help='BERTScore threshold considered low (generation failure)')
    parser.add_argument('--gen-min-len', type=int, default=10, help='Minimum generated answer length (chars)')
    parser.add_argument('--ctx-prec-low', type=float, default=0.05, help='Contextual precision threshold considered low')
    parser.add_argument('--halluc-high', type=float, default=0.3, help='Hallucination fraction threshold considered high')
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    records = load_json(args.metrics)
    qa_map = load_qa_types(args.qa_file)

    thresholds = {
        'bert_low': args.bert_low,
        'gen_min_len': args.gen_min_len,
        'ctx_prec_low': args.ctx_prec_low,
        'halluc_high': args.halluc_high,
    }

    per_type_counts, per_type_total, per_question_results = aggregate_by_type(records, qa_map, thresholds)
    summary, csv_file = save_outputs(args.out_dir, per_type_counts, per_type_total, per_question_results)
    plot_results(args.out_dir, per_type_counts, per_type_total)

    print('\n=== Summary ===')
    for t in sorted(per_type_total.keys()):
        print(f"{t}: total={per_type_total[t]}, {dict(per_type_counts[t])}")


if __name__ == '__main__':
    main()
