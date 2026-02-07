"""Run full RAG evaluation pipeline in one command.

Steps performed:
1. Load QA list from JSON.
2. Optionally run generation (uses `ResponseGenerator`) to produce `generated_answer`.
3. Compute metrics via `src.compute_metrics.compute_metrics` and save `data/metrics_per_question.json` and `data/metrics_summary.json`.
4. Run visualizations (`src.visualizations`) and error analysis (`src.error_analysis`).
5. Produce an HTML report and (optionally) a PDF report bundling key plots and tables.

Usage:
    python src/run_pipeline.py --qa-file data/qa_generated_100.json --out data/report --generate

Notes:
- Generation via `ResponseGenerator` can be slow and requires model/tokenizer dependencies.
- PDF export requires `weasyprint` or `pdfkit` (best-effort fallback to HTML only).
"""
import argparse
import datetime
import json
import os
import sys
import time
from typing import List, Dict

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rrf_fusion import RRFFuser
from src import compute_metrics
from src import visualizations
from src import error_analysis

# Optional generation module
try:
    from src.response_generation import ResponseGenerator
    GEN_AVAILABLE = True
except Exception:
    ResponseGenerator = None
    GEN_AVAILABLE = False

# Optional PDF conversion
WEASYPRINT_AVAILABLE = False
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_generation_for_list(qa_list: List[Dict], out_file: str, top_n: int = 5) -> List[Dict]:
    """Run ResponseGenerator for each question and attach `generated_answer` and `response_time`.

    Returns updated qa_list and writes `out_file`.
    """
    if not GEN_AVAILABLE:
        raise RuntimeError('Generation not available: ResponseGenerator failed to import or models missing')

    gen = ResponseGenerator()
    updated = []
    for i, qa in enumerate(qa_list, 1):
        q = qa.get('question') or qa.get('Question') or ''
        try:
            res = gen.generate(q, top_n=top_n)
            qa_out = qa.copy()
            qa_out['generated_answer'] = res.get('answer', '')
            qa_out['response_time'] = res.get('response_time', '')
            qa_out['fused_chunks'] = res.get('fused_chunks', [])
            updated.append(qa_out)
        except Exception as e:
            qa_out = qa.copy()
            qa_out['generated_answer'] = ''
            qa_out['response_time'] = ''
            updated.append(qa_out)
        if i % 10 == 0:
            print(f"Generated for {i}/{len(qa_list)} questions")
    # Save
    with open(out_file, 'w', encoding='utf-8') as wf:
        json.dump(updated, wf, indent=2, ensure_ascii=False)
    return updated


def write_json(path: str, obj):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=2, ensure_ascii=False)


def generate_html_report(summary: Dict, per_question: List[Dict], plots_dir: str, out_html: str):
    """Simple HTML report embedding key metrics and images from `plots_dir`."""
    now = datetime.datetime.utcnow().isoformat()
    # Build minimal HTML
    html_lines = [
        '<!doctype html>',
        '<html><head><meta charset="utf-8"><title>RAG Evaluation Report</title></head><body>',
        f'<h1>RAG Evaluation Report</h1>',
        f'<p>Generated: {now} UTC</p>',
        '<h2>Summary Metrics</h2>',
        '<ul>'
    ]
    for k, v in summary.items():
        if k == 'per_question':
            continue
        html_lines.append(f'<li><strong>{k}</strong>: {v}</li>')
    html_lines.append('</ul>')

    # Embed plots if present
    imgs = [f for f in os.listdir(plots_dir) if f.lower().endswith('.png')]
    if imgs:
        html_lines.append('<h2>Plots</h2>')
        for img in sorted(imgs):
            p = os.path.join(os.path.basename(plots_dir), img)
            html_lines.append(f'<h3>{img}</h3>')
            html_lines.append(f'<img src="{p}" style="max-width:900px;width:100%;height:auto;"/>')

    # Per-question table (first N rows)
    html_lines.append('<h2>Per-question (first 50)</h2>')
    html_lines.append('<table border="1" cellpadding="4" cellspacing="0">')
    # header
    headers = ['#', 'question', 'qa_type', 'found_rank', 'mrr_score', 'bertscore_f1', 'contextual_precision', 'hallucination_fraction']
    html_lines.append('<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>')
    for i, r in enumerate(per_question[:50], 1):
        row = [str(i), r.get('question','')[:200].replace('<','&lt;'), r.get('qa_type',''), str(r.get('found_rank','')), f"{r.get('mrr_score',0):.3f}", f"{r.get('bertscore_f1',0):.3f}", f"{r.get('contextual_precision',0):.3f}", f"{r.get('hallucination_fraction',0):.3f}"]
        html_lines.append('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>')
    html_lines.append('</table>')

    html_lines.append('</body></html>')
    html = '\n'.join(html_lines)
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    return out_html


def try_make_pdf(html_path: str, pdf_path: str) -> bool:
    if WEASYPRINT_AVAILABLE:
        try:
            weasyprint.HTML(html_path).write_pdf(pdf_path)
            return True
        except Exception:
            return False
    # fallback: no PDF
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-file', type=str, required=True, help='Input QA JSON file (list of questions)')
    parser.add_argument('--out-dir', type=str, default='data/report', help='Output directory for reports and artifacts')
    parser.add_argument('--generate', action='store_true', help='Run LLM generation for each question (slow)')
    parser.add_argument('--top-n', type=int, default=5, help='Top N chunks to send to generator')
    parser.add_argument('--k-per', type=int, default=50, help='Top-K per method for fusion when computing metrics')
    parser.add_argument('--n-final', type=int, default=100, help='Number of fused results to consider for metrics')
    parser.add_argument('--plots', action='store_true', help='Run visualizations')
    parser.add_argument('--adversarial', action='store_true', help='Run adversarial tests (if script present)')
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    # 1. Load QA
    with open(args.qa_file, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)

    gen_file = os.path.join(args.out_dir, 'qa_with_generated.json')
    if args.generate:
        if not GEN_AVAILABLE:
            print('Generation not available; skipping generation step')
        else:
            print('Running generation for questions (this may take a long time)...')
            qa_list = run_generation_for_list(qa_list, gen_file, top_n=args.top_n)
    else:
        # attach empty generated_answer if missing
        for qa in qa_list:
            if 'generated_answer' not in qa:
                qa['generated_answer'] = ''

    # 2. Compute metrics using existing compute_metrics.compute_metrics
    print('Computing metrics...')
    metrics = compute_metrics.compute_metrics(qa_list, k_per=args.k_per, n_final=args.n_final)
    metrics_out = os.path.join(args.out_dir, 'metrics_summary.json')
    per_q_out = os.path.join(args.out_dir, 'metrics_per_question.json')
    write_json(metrics_out, {k: v for k, v in metrics.items() if k != 'per_question'})
    write_json(per_q_out, metrics.get('per_question', []))
    print('Metrics written to', metrics_out, per_q_out)

    # 3. Visualizations: create plots summarizing retrieval and contextual metrics
    plots_dir = os.path.join(args.out_dir, 'plots')
    ensure_dir(plots_dir)
    if args.plots:
        print('Running visualizations...')
        # call visualizations functions programmatically and guard against signature mismatches
        try:
            visualizations.plot_metric_comparison(metrics.get('per_question', []), plots_dir)
        except Exception as e:
            print('Warning: failed to plot metric comparison:', e)

        try:
            visualizations.plot_score_distributions(metrics.get('per_question', []), plots_dir)
        except Exception as e:
            print('Warning: failed to plot score distributions:', e)

        try:
            # heatmap expects (records, out_dir, ...)
            visualizations.plot_retrieval_heatmap(metrics.get('per_question', []), plots_dir)
        except TypeError:
            # older signatures may require different ordering; try fallback with explicit kwarg
            try:
                visualizations.plot_retrieval_heatmap(records=metrics.get('per_question', []), out_dir=plots_dir)
            except Exception as e:
                print('Warning: failed to plot retrieval heatmap:', e)
        except Exception as e:
            print('Warning: failed to plot retrieval heatmap:', e)

        try:
            visualizations.plot_ablation_results(os.path.join('data', 'ablation_results.json'), plots_dir)
        except Exception as e:
            print('Warning: failed to plot ablation results:', e)
    else:
        print('Skipping visualizations (use --plots to enable)')

    # 4. Error analysis
    print('Running error analysis...')
    # error_analysis expects metrics file path and optional qa-file mapping; call programmatically
    # reuse functions: aggregate_by_type and plotting
    qa_map = {}  # not passing qa-file mapping here
    thresholds = {'bert_low': 0.15, 'gen_min_len': 10, 'ctx_prec_low': 0.05, 'halluc_high': 0.3}
    per_type_counts, per_type_total, per_question_results = error_analysis.aggregate_by_type(metrics.get('per_question', []), qa_map, thresholds)
    summary, csv_file = error_analysis.save_outputs(args.out_dir, per_type_counts, per_type_total, per_question_results)
    # produce error plots into plots_dir if matplotlib available
    try:
        error_analysis.plot_results(args.out_dir, per_type_counts, per_type_total)
    except Exception:
        pass

    # 5. Optional adversarial (call script if present)
    if args.adversarial:
        adv_script = os.path.join(PROJECT_ROOT, 'src', 'adversarial_testing.py')
        if os.path.exists(adv_script):
            print('Running adversarial tests...')
            # Best-effort: spawn as subprocess to avoid heavy imports here
            import subprocess
            subprocess.run([sys.executable, adv_script, '--out', os.path.join(args.out_dir, 'adversarial_results.json')])
        else:
            print('No adversarial script found; skipping')

    # 6. Visual report
    print('Generating HTML report...')
    html_path = os.path.join(args.out_dir, 'report.html')
    generate_html_report(metrics, metrics.get('per_question', []), plots_dir, html_path)

    pdf_path = os.path.join(args.out_dir, 'report.pdf')
    if try_make_pdf(html_path, pdf_path):
        print('PDF report generated at', pdf_path)
    else:
        print('PDF generation not available or failed; HTML report at', html_path)

    # 7. Save CSV of per-question metrics
    csv_out = os.path.join(args.out_dir, 'metrics_per_question.csv')
    with open(csv_out, 'w', encoding='utf-8', newline='') as cf:
        import csv
        writer = csv.writer(cf)
        headers = ['question', 'qa_type', 'found_rank', 'mrr_score', 'bertscore_f1', 'contextual_precision', 'hallucination_fraction', 'generated_answer']
        writer.writerow(headers)
        for r in metrics.get('per_question', []):
            writer.writerow([
                r.get('question','')[:500],
                r.get('qa_type',''),
                r.get('found_rank',''),
                r.get('mrr_score',''),
                r.get('bertscore_f1',''),
                r.get('contextual_precision',''),
                r.get('hallucination_fraction',''),
                (r.get('generated_answer') or '')[:500]
            ])
    print('CSV written to', csv_out)

    print('\nPipeline complete. Outputs in:', args.out_dir)


if __name__ == '__main__':
    main()
