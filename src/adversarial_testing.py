"""Adversarial testing harness for Hybrid RAG.

Generates challenging question types (ambiguous, negated, multi-hop), creates
paraphrases, and constructs unanswerable questions to probe retrieval and
hallucination behaviour. Runs the `RRFFuser` to obtain fused context and
computes simple heuristics (entity extraneousness) to flag potential
hallucinations.

Usage:
    python src/adversarial_testing.py --num 20 --out data/adversarial_results.json

The script is lightweight and does not invoke heavy LLM generation by default.
If a paraphrase pipeline is available, it will be used; otherwise a simple
rule-based paraphrase is applied.
"""
import argparse
import json
import os
import random
import sys
from typing import List, Dict

# Make project root importable (matches other scripts)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rrf_fusion import RRFFuser
from src.compute_metrics import _extract_entities, _tokenize_simple, normalize_url

try:
    # Optional paraphrase pipeline (best-effort)
    from transformers import pipeline
    PARA_PIPELINE = None
    try:
        PARA_PIPELINE = pipeline('text2text-generation', model='Vamsi/T5_Paraphrase_Paws', device=-1)
    except Exception:
        try:
            PARA_PIPELINE = pipeline('text2text-generation', model='t5-base', device=-1)
        except Exception:
            PARA_PIPELINE = None
except Exception:
    PARA_PIPELINE = None


def load_titles_and_snippets(chunks_file: str) -> List[Dict]:
    """Load chunk metadata (title, url, snippet) used to craft adversarial Qs."""
    if not os.path.exists(chunks_file):
        return []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks


def make_ambiguous_question(title: str) -> str:
    """Produce an ambiguous question referencing a pronoun or vague phrase."""
    return f"When did it happen with respect to {title}?"


def make_negation_question(title: str) -> str:
    """Produce a negation-style question that tests retrieval for exclusions."""
    return f"Which event related to {title} did NOT occur in the 20th century?"


def make_multihop_question(title1: str, title2: str) -> str:
    """Produce a simple multi-hop question combining two page titles."""
    return f"How did {title1} influence the development of {title2}?"


def make_unanswerable_question(title: str) -> str:
    """Create an unanswerable question (asks for false detail or unrelated fact)."""
    return f"What is the population of the fictional city of {title} in 1850?"


def paraphrase_question(q: str) -> str:
    """Paraphrase question using pipeline if available, else apply light rules."""
    if PARA_PIPELINE is not None:
        try:
            out = PARA_PIPELINE(q, max_length=128, num_return_sequences=1)
            text = out[0].get('generated_text') if isinstance(out[0], dict) else str(out[0])
            # Strip any leading prompt echo
            if text.startswith(q):
                text = text[len(q):].strip()
            text = text.strip()
            if text:
                return text
        except Exception:
            pass
    # Fallback: simple syntactic variation
    q2 = q.replace('Which', 'Which one').replace('How did', 'In what way did').replace('What is', 'Can you tell me')
    return q2


def craft_adversarial_set(chunks: List[Dict], num_per_type: int = 5) -> List[Dict]:
    """Create adversarial question list from available chunks.

    Returns list of dicts: {'question', 'type', 'ground_truth' (if available)}
    """
    rnd = random.Random(42)
    pool = chunks[:] if chunks else []
    rnd.shuffle(pool)
    questions = []

    # Use titles from chunks
    titles = [c.get('title') or c.get('chunk_id') for c in pool if c.get('title')]
    if not titles:
        titles = [f"Entity{i}" for i in range(50)]

    # Ambiguous: pronoun/vague questions to test system's handling of referents
    for i in range(min(num_per_type, len(titles))):
        t = titles[i]
        q = make_ambiguous_question(t)
        questions.append({'question': q, 'type': 'ambiguous', 'seed_titles': [t]})

    # Negation
    for i in range(min(num_per_type, len(titles))):
        t = titles[(i + 7) % len(titles)]
        q = make_negation_question(t)
        questions.append({'question': q, 'type': 'negation', 'seed_titles': [t]})

    # Multi-hop
    for i in range(min(num_per_type, len(titles) - 1)):
        t1 = titles[i]
        t2 = titles[(i + 3) % len(titles)]
        q = make_multihop_question(t1, t2)
        questions.append({'question': q, 'type': 'multihop', 'seed_titles': [t1, t2]})

    # Paraphrase robustness (create base question + a paraphrased variant)
    for i in range(min(num_per_type, len(titles))):
        t = titles[(i + 11) % len(titles)]
        base = f"What are the main contributions of {t}?"
        para = paraphrase_question(base)
        questions.append({'question': base, 'type': 'paraphrase_base', 'seed_titles': [t]})
        questions.append({'question': para, 'type': 'paraphrase_para', 'seed_titles': [t]})

    # Unanswerable
    for i in range(min(num_per_type, len(titles))):
        t = titles[(i + 17) % len(titles)]
        q = make_unanswerable_question(t)
        questions.append({'question': q, 'type': 'unanswerable', 'seed_titles': [t]})

    return questions


def run_adversarial_tests(questions: List[Dict], out_file: str, k_per: int = 50, n_final: int = 20):
    fuser = RRFFuser()
    results = []
    for i, item in enumerate(questions, 1):
        q = item['question']
        qtype = item.get('type', '')
        fused = fuser.fuse(q, k_per=k_per, n_final=n_final)

        # Concatenate retrieved context
        retrieved_texts = [c.get('full_text') or c.get('text') or '' for c in fused]
        retrieved_context = '\n'.join(retrieved_texts)

        # Entities
        ret_ents = _extract_entities(retrieved_context)

        # For adversarial tests we usually don't have ground-truth answers; use seed_titles as proxy
        seed_titles = item.get('seed_titles', [])
        gt_entities = set(seed_titles)

        extraneous = [e for e in ret_ents if e not in gt_entities]
        halluc_frac = len(extraneous) / max(1, len(ret_ents)) if ret_ents else 0.0

        # Basic URL coverage: check if any seed page titles appear in retrieved URLs (simple proxy)
        found_seed_urls = []
        for c in fused:
            url = normalize_url(c.get('url') or '')
            for st in seed_titles:
                if st and st.lower().replace(' ', '_') in url:
                    found_seed_urls.append(st)

        results.append({
            'idx': i,
            'question': q,
            'type': qtype,
            'seed_titles': seed_titles,
            'num_retrieved': len(fused),
            'retrieved_titles': [c.get('title') for c in fused],
            'retrieved_urls': [c.get('url') for c in fused],
            'retrieved_entities': list(ret_ents),
            'extraneous_entities': extraneous,
            'hallucination_fraction': halluc_frac,
            'found_seed_titles_in_urls': list(set(found_seed_urls))
        })

    # Save results
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as wf:
        json.dump(results, wf, indent=2, ensure_ascii=False)
    print(f"Adversarial results written to: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=5, help='Num per adversarial type')
    parser.add_argument('--out', type=str, default='data/adversarial_results.json')
    parser.add_argument('--chunks', type=str, default='data/chunks.json')
    parser.add_argument('--k-per', type=int, default=50)
    parser.add_argument('--n-final', type=int, default=20)
    args = parser.parse_args()

    chunks = load_titles_and_snippets(args.chunks)
    questions = craft_adversarial_set(chunks, num_per_type=args.num)
    print(f"Crafted {len(questions)} adversarial questions (types: ambiguous, negation, multihop, paraphrase, unanswerable)")
    run_adversarial_tests(questions, args.out, k_per=args.k_per, n_final=args.n_final)


if __name__ == '__main__':
    main()
