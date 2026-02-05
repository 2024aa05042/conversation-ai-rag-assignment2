"""Compute MRR, BERTScore, Contextual Precision, and Hallucination Detection (URL-level).

Metrics implemented and justification:
- MRR: Mean Reciprocal Rank at URL level — how quickly the system retrieves the correct source.
    Important for retrieval-focused RAG evaluation because it directly measures ranking effectiveness.

- BERTScore (semantic overlap between retrieved context and ground-truth snippet):
    - Justification: BERTScore uses contextual embeddings to measure semantic similarity beyond
        token overlap. It is robust to paraphrase and measures whether the retrieved passages
        semantically match the ground-truth evidence — a proxy for retrieval usefulness for answer
        generation.
    - Calculation: uses `bert_score.score(cands, refs, lang='en')` if installed; otherwise falls
        back to sentence-transformers cosine similarity as a proxy.

- Contextual Precision (token-level overlap of retrieved context with ground-truth):
    - Justification: measures how much of the retrieved context is actually relevant (precision
        of retrieved tokens w.r.t. ground-truth). High contextual precision means less noise to the
        generator and fewer distracting tokens.
    - Calculation: tokens(retrieved) ∩ tokens(gt) / tokens(retrieved).

- Hallucination Detection (heuristic entity-level extraneousness):
    - Justification: approximates whether the retrieved context introduces entities or named
        concepts not supported by the ground-truth snippet — a proxy for potential hallucinations
        if the generator uses unsupported claims.
    - Calculation: extract entity candidates (spaCy if available, else simple capitalization regex),
        count entities present in retrieved context but not in ground-truth, normalize by total
        retrieved entities.

Usage:
    python scripts/compute_metrics.py --qa-file data/qa_generated_100.json --k-list 1,5,10 --k-per 50
"""
import argparse
import json
import math
import os
import sys

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rrf_fusion import RRFFuser

# Optional high-quality metric backends
try:
    from bert_score import score as bert_score_score
    BERTSCORE_AVAILABLE = True
except Exception:
    bert_score_score = None
    BERTSCORE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ST_AVAILABLE = True
    _st_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    ST_AVAILABLE = False
    _st_model = None

try:
    import spacy
    try:
        _nlp = spacy.load('en_core_web_sm')
        SPACY_AVAILABLE = True
    except Exception:
        _nlp = None
        SPACY_AVAILABLE = False
except Exception:
    spacy = None
    _nlp = None
    SPACY_AVAILABLE = False


def normalize_url(u: str) -> str:
    if not u:
        return ''
    return u.rstrip('/').lower()


def _tokenize_simple(text: str):
    # Lowercase, split on non-word
    import re
    toks = re.findall(r"\w+", (text or '').lower())
    return toks


def _extract_entities(text: str):
    # Prefer spaCy NER when available
    ents = set()
    if not text:
        return ents
    if SPACY_AVAILABLE and _nlp is not None:
        doc = _nlp(text)
        for e in doc.ents:
            ents.add(e.text.strip())
    else:
        # Fallback: capitalized multi-word sequences + capitalized single tokens
        import re
        for m in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            ents.add(m.strip())
    return ents


def compute_metrics(qa_list, k_per=50, n_final=100, k_list=(1, 5, 10)):
    """Compute MRR plus BERTScore, Contextual Precision, and Hallucination heuristics.

    For each QA pair:
    - Use `RRFFuser.fuse` to obtain fused top `n_final` chunks and concatenate their `full_text`
      (this is the retrieved context).
    - MRR: as before (URL-level rank of true source).
    - BERTScore: semantic similarity between retrieved_context (candidate) and `ground_truth` (reference).
    - Contextual Precision: token overlap fraction = |tokens(retrieved) ∩ tokens(gt)| / |tokens(retrieved)|.
    - Hallucination heuristic: entities in retrieved_context not present in ground-truth,
      hallucination_fraction = |extraneous_entities| / max(1, |entities_in_retrieved|).
    """
    fuser = RRFFuser()
    Q = len(qa_list)

    reciprocal_sum = 0.0
    bert_f1_sum = 0.0
    ctx_precision_sum = 0.0
    halluc_frac_sum = 0.0
    per_question = []

    # For fallback embedding-based BERTScore proxy
    def bertscore_proxy(cands, refs):
        if BERTSCORE_AVAILABLE and bert_score_score is not None:
            P, R, F1 = bert_score_score(cands, refs, lang='en', rescale_with_baseline=True)
            return [float(x) for x in F1]
        elif ST_AVAILABLE and _st_model is not None:
            cand_emb = _st_model.encode(cands, convert_to_numpy=True, normalize_embeddings=True)
            ref_emb = _st_model.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
            sims = (cosine_similarity(cand_emb, ref_emb).diagonal()).tolist()
            # map cosine sim [-1,1] -> [0,1]
            return [(s + 1) / 2 for s in sims]
        else:
            # Last resort: token overlap as weak proxy
            scores = []
            for c, r in zip(cands, refs):
                ct = set(_tokenize_simple(c))
                rt = set(_tokenize_simple(r))
                scores.append(len(ct & rt) / max(1, len(rt)))
            return scores

    for i, qa in enumerate(qa_list, 1):
        q = qa.get('question') or qa.get('Question') or ''
        true_url = normalize_url(qa.get('url'))
        ground_truth = qa.get('ground_truth') or qa.get('answer') or ''

        fused = fuser.fuse(q, k_per, n_final)

        # Build ranked URL list (unique, preserve order)
        seen = set()
        ranked_urls = []
        retrieved_texts = []
        for item in fused:
            url = normalize_url(item.get('url', ''))
            if url and url not in seen:
                seen.add(url)
                ranked_urls.append(url)
            # collect full_text for contextual metrics
            ft = item.get('full_text') or item.get('text') or ''
            retrieved_texts.append(ft)

        # Concatenate retrieved context (limit length to avoid huge inputs)
        retrieved_context = '\n'.join(retrieved_texts)[:4000]

        # MRR rank
        rank = None
        for idx, url in enumerate(ranked_urls, 1):
            if url == true_url:
                rank = idx
                break
        if rank:
            reciprocal_sum += 1.0 / rank
        # per-question MRR contribution (1/rank or 0)
        per_mrr = 1.0 / rank if rank else 0.0

        # BERTScore (or proxy) between retrieved_context (candidate) and ground_truth (reference)
        bert_scores = bertscore_proxy([retrieved_context], [ground_truth])
        bert_f1 = bert_scores[0]
        bert_f1_sum += bert_f1

        # Contextual Precision: token overlap fraction
        ret_toks = _tokenize_simple(retrieved_context)
        gt_toks = _tokenize_simple(ground_truth)
        if len(ret_toks) == 0:
            ctx_prec = 0.0
        else:
            ctx_prec = len(set(ret_toks) & set(gt_toks)) / len(ret_toks)
        ctx_precision_sum += ctx_prec

        # Hallucination heuristic: entities in retrieved but not in ground-truth
        ret_ents = _extract_entities(retrieved_context)
        gt_ents = _extract_entities(ground_truth)
        if len(ret_ents) == 0:
            halluc_frac = 0.0
        else:
            extraneous = [e for e in ret_ents if e not in gt_ents]
            halluc_frac = len(extraneous) / max(1, len(ret_ents))
        halluc_frac_sum += halluc_frac

        per_question.append({
            'question': q,
            'ground_truth': ground_truth,
            'generated_answer': qa.get('generated_answer') or qa.get('answer') or '',
            'true_url': true_url,
            'found_rank': rank,
            'mrr_score': per_mrr,
            'bertscore_f1': bert_f1,
            'contextual_precision': ctx_prec,
            'hallucination_fraction': halluc_frac
        })

    # Aggregate averages
    mrr = reciprocal_sum / Q if Q else 0.0
    avg_bert_f1 = bert_f1_sum / Q if Q else 0.0
    avg_ctx_prec = ctx_precision_sum / Q if Q else 0.0
    avg_halluc = halluc_frac_sum / Q if Q else 0.0

    return {
        'Q': Q,
        'MRR': mrr,
        'BERTScore_F1': avg_bert_f1,
        'Contextual_Precision': avg_ctx_prec,
        'Hallucination_Fraction': avg_halluc,
        'per_question': per_question
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-file', type=str, default='data/qa_generated_100.json')
    parser.add_argument('--k-per', type=int, default=50, help='Top-K per method to retrieve before fusion')
    parser.add_argument('--n-final', type=int, default=100, help='Number of final fused results to consider')
    parser.add_argument('--k-list', type=str, default='1,5,10', help='Comma-separated K values for @K metrics')
    args = parser.parse_args()

    if not os.path.exists(args.qa_file):
        raise FileNotFoundError(f"QA file not found: {args.qa_file}")

    with open(args.qa_file, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)

    print(f"Running metrics on {len(qa_list)} QA pairs; k_per={args.k_per}")
    results = compute_metrics(qa_list, k_per=args.k_per, n_final=args.n_final)

    print('\n=== Retrieval + Context Metrics (URL-level) ===')
    print(f"Q = {results['Q']}")
    print(f"MRR = {results['MRR']:.4f}  (higher is better; 1.0 means perfect ranking)")
    print(f"BERTScore F1 (avg) = {results['BERTScore_F1']:.4f}  (semantic overlap between retrieved context and ground-truth)")
    print(f"Contextual Precision = {results['Contextual_Precision']:.4f}  (fraction of retrieved tokens that overlap ground-truth; higher is better)")
    print(f"Hallucination Fraction = {results['Hallucination_Fraction']:.4f}  (avg fraction of retrieved entities not in ground-truth; lower is better)")

    # Save per-question metrics to data/metrics_per_question.json
    try:
        DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    except Exception:
        DATA_DIR = 'data'

    out_file = os.path.join(DATA_DIR, 'metrics_per_question.json')
    try:
        with open(out_file, 'w', encoding='utf-8') as wf:
            json.dump(results.get('per_question', []), wf, indent=2, ensure_ascii=False)
        print(f"Per-question metrics written to: {out_file}")
    except Exception as e:
        print(f"Failed to write per-question metrics: {e}")
    # Print tabular summary (Question, Ground Truth, Generated Answer, MRR, BERT, CtxPrec, Halluc)
    rows = results.get('per_question', [])
    if rows:
        def trunc(s, n):
            s = (s or '')
            s = s.replace('\n', ' ')
            return s if len(s) <= n else s[:n-3] + '...'

        qw, gw, aw = 60, 60, 60
        hdr = f"{'#':>3} | {'Question':{qw}} | {'Ground Truth':{gw}} | {'Generated Answer':{aw}} | {'MRR':>5} | {'BERT':>6} | {'CtxP':>5} | {'Halluc':>6}"
        sep = '-' * len(hdr)
        print('\n' + hdr)
        print(sep)
        for i, r in enumerate(rows, 1):
            qtxt = trunc(r.get('question',''), qw)
            gtxt = trunc(r.get('ground_truth',''), gw)
            atxt = trunc(r.get('generated_answer',''), aw)
            mrr_v = r.get('mrr_score', 0.0)
            bert_v = r.get('bertscore_f1', 0.0)
            ctx_v = r.get('contextual_precision', 0.0)
            hal_v = r.get('hallucination_fraction', 0.0)
            print(f"{i:3d} | {qtxt:{qw}} | {gtxt:{gw}} | {atxt:{aw}} | {mrr_v:5.3f} | {bert_v:6.3f} | {ctx_v:5.3f} | {hal_v:6.3f}")


if __name__ == '__main__':
    main()
