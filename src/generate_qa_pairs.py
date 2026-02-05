"""
Generate 100 diverse QA pairs from the fixed Wikipedia URLs listed in `data/fixedurls.json`.

Behavior summary:
- Fetches and caches full page texts for the fixed URLs at `data/fixed_pages.json` (use --fetch to refresh).
- For each page it tries to create one QA pair per type: factual, comparative, inferential, multihop.
- If a page fails to produce valid QA for any required type, the page is skipped.
- Continues until `--num` QA pairs are produced or pages are exhausted.

Outputs:
- `data/qa_generated_<N>.json` and `data/qa_generated_<N>.jsonl` (default N=100)

Usage:
    python scripts/generate_qa_pairs.py --num 100 --model google/flan-t5-small --fetch

Note: This script requires `requests` and `beautifulsoup4` (add to venv if missing).
"""

import os
import json
import random
import argparse
import time
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# CONFIG
NUM_QA_TOTAL = 4
QA_TYPES = ["factual", "comparative", "inferential", "multihop"]
FIXED_URLS_FILE = os.path.join("data", "fixedurls.json")
CHUNKS_FILE = os.path.join("data", "chunks.json")
FIXED_PAGES_CACHE = os.path.join("data", "fixed_pages.json")


def load_fixed_urls(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        urls = json.load(f)
    return urls


def fetch_wikipedia_page(url: str, session: requests.Session, pause: float = 1.0) -> Dict:
    headers = {"User-Agent": "Hybrid-RAG-QA-Generator/1.0"}
    try:
        resp = session.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return {'url': url, 'title': '', 'text': ''}
        soup = BeautifulSoup(resp.text, 'html.parser')
        title_tag = soup.find('h1', id='firstHeading')
        title = title_tag.get_text(strip=True) if title_tag else url
        content_div = soup.find('div', id='mw-content-text') or soup.find('div', class_='mw-parser-output')
        paras = []
        if content_div:
            for p in content_div.find_all('p'):
                txt = p.get_text(separator=' ', strip=True)
                if len(txt) > 40:
                    paras.append(txt)
        text = '\n\n'.join(paras)
        time.sleep(pause)
        return {'url': url, 'title': title, 'text': text}
    except Exception:
        return {'url': url, 'title': '', 'text': ''}


def load_or_fetch_fixed_pages(urls: List[str], cache_path: str, fetch: bool = False) -> List[Dict]:
    if os.path.exists(cache_path) and not fetch:
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    session = requests.Session()
    pages = []
    for i, url in enumerate(urls):
        print(f"Fetching ({i+1}/{len(urls)}): {url}")
        page = fetch_wikipedia_page(url, session)
        pages.append(page)

    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return pages


def sentence_split(text: str) -> List[str]:
    s = text.replace('\n', ' ')
    candidates = [sent.strip() for sent in s.split('.') if len(sent.strip()) > 30]
    return candidates


def select_answer_sentence(doc_text: str) -> str:
    sents = sentence_split(doc_text)
    if not sents:
        return doc_text.strip()[:200]
    for sent in sents:
        if any(ch.isdigit() for ch in sent):
            return sent
    return random.choice(sents)


def make_prompt(qtype: str, answer: str, context: str) -> str:
    # Stronger, explicit prompt templates to encourage single-sentence, grammatical questions
    if qtype == 'factual':
        return (
            f"You are a bot who is responsible to generate a relevant factual question from a given answer extracted from a context. Write ONE clear, grammatically correct factual question (single sentence) whose answer is the given Answer.\n"
            f"Context: {context}\nAnswer: {answer}\nQuestion (single sentence):"
        )
    if qtype == 'comparative':
        return (
            f"You are a bot who is responsible to generate a relevant comparative question from a given answer extracted from a context. Write ONE clear, grammatically correct comparative question (single sentence) that asks to compare or contrast aspects in the context.\n"
            f"Give a single concise question. Example: 'How does X compare to Y in terms of Z?'\n"
            f"Context: {context}\nFocus: {answer}\nQuestion (single sentence):"
        )
    if qtype == 'inferential':
        return (
            f"You are a bot who is responsible to generate a relevant inferential question from a given answer extracted from a context. Write ONE clear, grammatically correct inferential question (single sentence) that asks for a reason or explanation based on the context.\n"
            f"Give a single concise question. Example: 'Why did X happen?' or 'What explains X?'\n"
            f"Context: {context}\nAnswer hint: {answer}\nQuestion (single sentence):"
        )
    if qtype == 'multihop':
        return (
            f"You are a bot who is responsible to generate a relevant multi-hop question from a given answer extracted from a context. Write ONE clear, grammatically correct multi-hop question (single sentence) that requires combining multiple facts from the context.\n"
            f"Give a single concise question. Example: 'How did A lead to B through C?'\n"
            f"Context: {context}\nTarget fact: {answer}\nQuestion (single sentence):"
        )
    return (
        f"You are a bot who is responsible to generate a relevant question from a given answer extracted from a context.Write ONE clear, grammatical question (single sentence).\nContext: {context}\nAnswer: {answer}\nQuestion (single sentence):"
    )


def init_generator(model_name: str = 'mrm8488/t5-base-finetuned-question-generation-ap'):
    print(f"Loading generator model: {model_name} (may download)...")
    # Some transformer versions don't expose 'text2text-generation'. Try it first,
    # then fall back to 'text-generation' for compatibility.
    try:
        return pipeline('text2text-generation', model=model_name, device=-1)
    except Exception:
        print("'text2text-generation' pipeline not available, falling back to 'text-generation'")
        return pipeline('text-generation', model=model_name, device=-1)


def validate_question_for_type(q: str, qtype: str) -> bool:
    ql = q.lower()
    if len(q.strip()) < 8:
        return False
    if not q.strip().endswith('?'):
        return False
    
    if qtype == 'comparative':
        keywords = ['compare', 'than', 'vs', 'difference', 'compare', 'differences', 'rather than']
        return any(k in ql for k in keywords)
    if qtype == 'inferential':
        return any(w in ql for w in ['why', 'how', 'reason', 'explain'])
    if qtype == 'multihop':
        # heuristic: ask for how or which and contain 'and' or 'then'
        return any(w in ql for w in ['how', 'which', 'why']) and (' and ' in ql or ' then ' in ql or ',' in ql)
    # factual
    return True


def clean_text_basic(s: str) -> str:
    import re
    s = re.sub(r"[\r\t\x0b\x0c]", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s


def is_garbage_text(q: str) -> bool:
    if not q:
        return True
    letters = sum(1 for ch in q if ch.isalpha())
    return (letters / max(1, len(q))) < 0.5


def generate_question(generator, prompt: str, max_length: int = 80) -> str:
    import re

    def clean_text(s: str) -> str:
        # remove non-printable control chars, excessive spaces
        s = re.sub(r"[\r\t\x0b\x0c]", ' ', s)
        s = re.sub(r"\s+", ' ', s).strip()
        return s

    def is_garbage(q: str) -> bool:
        # ratio of alphabetic characters
        if not q:
            return True
        letters = sum(1 for ch in q if ch.isalpha())
        return (letters / max(1, len(q))) < 0.5

    # Try sampling first for diversity
    try_settings = [
        {'do_sample': True, 'top_k': 50, 'temperature': 0.7},
        {'do_sample': False, 'top_k': None, 'temperature': None}
    ]

    for settings in try_settings:
        try:
            kwargs = {'max_length': max_length, 'num_return_sequences': 1}
            if settings['do_sample']:
                kwargs.update({'do_sample': True, 'top_k': settings['top_k'], 'temperature': settings['temperature']})
            else:
                kwargs.update({'do_sample': False})
            out = generator(prompt, **kwargs)
            text = out[0].get('generated_text') if isinstance(out[0], dict) else str(out[0])
            # remove only the prompt prefix occurrence at the start
            if text.startswith(prompt):
                q = text[len(prompt):]
            else:
                q = text
            q = clean_text(q)
            if '\n' in q:
                q = q.split('\n')[0].strip()
            if not q.endswith('?'):
                # try to cut to first sentence
                if '?' in q:
                    q = q.split('?')[0].strip() + '?'
                else:
                    q = q.split('.')[0].strip() + '?'
            # Ensure capitalization
            if q and q[0].islower():
                q = q[0].upper() + q[1:]

            if not is_garbage(q):
                return q
        except Exception:
            continue

    # Fallback: create a safe, templated factual question from the answer
    a = prompt.split('Answer:')[-1].split('\n')[0].strip()
    short = ' '.join(a.split()[:12])
    if len(short) < 3:
        return ''
    fallback = f"What does the passage say about {short}?"
    # Clean fallback
    fallback = re.sub(r"\s+", ' ', fallback).strip()
    if not fallback.endswith('?'):
        fallback = fallback + '?'
    return fallback


def main():
    """Command-line entrypoint for QA pair generation.

    Loads fixed Wikipedia URLs, optionally fetches and caches page text, then
    generates balanced QA pairs by type (factual, comparative, inferential, multihop).
    Outputs JSON/JSONL files under `data/`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=NUM_QA_TOTAL)
    parser.add_argument('--model-factual', type=str, default='mrm8488/t5-base-finetuned-question-generation-ap',
                        help='Model for factual question generation')
    parser.add_argument('--model-comparative', type=str, default='google/flan-t5-large',
                        help='Model for comparative question generation')
    parser.add_argument('--model-inferential', type=str, default='google/flan-t5-large',
                        help='Model for inferential question generation')
    parser.add_argument('--model-multihop', type=str, default='google/flan-t5-large',
                        help='Model for multi-hop question generation')
    parser.add_argument('--model-rewriter', type=str, default='vennify/t5-base-grammar-correction',
                        help='Model used to rewrite/paraphrase generated questions for grammar')
    parser.add_argument('--fetch', action='store_true', help='Refetch fixed pages and refresh cache')
    args = parser.parse_args()

    num = args.num
    fixed_urls = load_fixed_urls(FIXED_URLS_FILE)
    pages = load_or_fetch_fixed_pages(fixed_urls, FIXED_PAGES_CACHE, fetch=args.fetch)

    # Build docs from fetched pages
    docs = [p for p in pages if p.get('text')]
    print(f"Using {len(docs)} fetched documents (from {len(fixed_urls)} URLs)")

    # Initialize per-type generators (lazy load)
    per_type_models = {
        'factual': args.model_factual,
        'comparative': args.model_comparative,
        'inferential': args.model_inferential,
        'multihop': args.model_multihop,
    }
    generators: Dict[str, object] = {}
    rewriter_model = args.model_rewriter
    rewriter_gen = None

    qa_list = []
    # target per type
    per_type_target = num // len(QA_TYPES)
    max_attempts_per_type = max(200, per_type_target * 8)

    # helper to get two sentences for comparative/multihop
    def pick_two_sentences(text: str):
        sents = sentence_split(text)
        if len(sents) >= 2:
            return random.sample(sents, 2)
        if sents:
            return (sents[0], sents[0])
        return (text[:120], text[:120])

    # For each QA type, independently generate `per_type_target` items
    for qtype in QA_TYPES:
        created = 0
        attempts = 0
        used_pages_for_type = set()
        print(f"\nGenerating up to {per_type_target} items for type: {qtype}")

        # shuffle docs for randomness
        doc_indices = list(range(len(docs)))
        while created < per_type_target and attempts < max_attempts_per_type:
            attempts += 1
            if not doc_indices:
                doc_indices = list(range(len(docs)))
                random.shuffle(doc_indices)

            idx = doc_indices.pop()
            p = docs[idx]
            page_id = p.get('title') or p.get('url')
            if page_id in used_pages_for_type:
                continue
            text = p.get('text', '')
            if not text or len(text) < 200:
                continue

            # choose answer(s) depending on type
            if qtype == 'factual' or qtype == 'inferential':
                answer = select_answer_sentence(text)
            elif qtype in ('comparative', 'multihop'):
                s1, s2 = pick_two_sentences(text)
                # encode both sentences as the 'answer' with delimiter for prompt usage
                answer = s1 + ' || ' + s2
            else:
                answer = select_answer_sentence(text)

            context = text[:1200]
            prompt = make_prompt(qtype, answer, context)
            # get or create generator for this type
            if qtype not in generators:
                try:
                    generators[qtype] = init_generator(per_type_models[qtype])
                except Exception as e:
                    print(f"Failed to load model for {qtype}: {e}")
                    continue
            gen = generators[qtype]
            try:
                question = generate_question(gen, prompt)
            except Exception as e:
                continue

            # basic cleaning
            question = clean_text_basic(question)
            # If the question looks like garbage or fails type validation, try a rewrite step
            if is_garbage_text(question) or not validate_question_for_type(question, qtype):
                # try a simple rewrite using the same generator (if it supports text2text)
                try:
                    # Prefer using a dedicated rewriter model if provided
                    rewrite_prompt = f"Paraphrase into ONE clear, grammatical question (single sentence): {question}"
                    if rewriter_model:
                        if rewriter_gen is None:
                            try:
                                rewriter_gen = init_generator(rewriter_model)
                            except Exception as e:
                                rewriter_gen = None
                                print(f"Failed to load rewriter model: {e}")
                        if rewriter_gen is not None:
                            rewrite_out = rewriter_gen(rewrite_prompt, max_length=80, do_sample=False)
                        else:
                            rewrite_out = generators[qtype](rewrite_prompt, max_length=80, do_sample=False)
                    else:
                        rewrite_out = generators[qtype](rewrite_prompt, max_length=80, do_sample=False)

                    rewritten = rewrite_out[0].get('generated_text') if isinstance(rewrite_out[0], dict) else str(rewrite_out[0])
                    if rewritten.startswith(rewrite_prompt):
                        rewritten = rewritten[len(rewrite_prompt):]
                    rewritten = clean_text_basic(rewritten)
                    if rewritten and not is_garbage_text(rewritten) and validate_question_for_type(rewritten, qtype):
                        question = rewritten
                except Exception:
                    pass

            if not question:
                continue

            if not validate_question_for_type(question, qtype):
                continue

            qa = {
                'question': question,
                'ground_truth': answer.strip(),
                'source_url': p.get('url'),
                'chunk_id': p.get('title')[:120],
                'doc_title': p.get('title', '')[:120],
                'qa_type': qtype
            }
            qa_list.append(qa)
            used_pages_for_type.add(page_id)
            created += 1
            if created % 5 == 0 or created < 5:
                print(f"  [{qtype}] created {created}/{per_type_target}")

        print(f"Finished {qtype}: created {created}/{per_type_target} after {attempts} attempts")

    # If still short, fall back to chunk-level generation from local chunks.json
    if len(qa_list) < num:
        print(f"Only generated {len(qa_list)} QA from pages; falling back to `chunks.json` to fill remaining.")
        try:
            with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        except Exception:
            chunks = []

        candidates = [c for c in chunks if c.get('text')]
        random.shuffle(candidates)
        for c in candidates:
            if len(qa_list) >= num:
                break
            ans = select_answer_sentence(c['text'])
            context = c['text'][:1000]
            q = make_prompt('factual', ans, context)
            try:
                question = generate_question(gen, q)
            except Exception:
                continue
            if not question:
                continue
            qa_list.append({
                'question': question,
                'ground_truth': ans.strip(),
                'source_url': c.get('url'),
                'chunk_id': c.get('chunk_id'),
                'doc_title': c.get('title', '')[:120],
                'qa_type': 'factual'
            })

    qa_list = qa_list[:num]

    out_json = os.path.join('data', f'qa_generated_{len(qa_list)}.json')
    out_jsonl = os.path.join('data', f'qa_generated_{len(qa_list)}.jsonl')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(qa_list, f, indent=2, ensure_ascii=False)
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for qa in qa_list:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')

    print(f"Saved {len(qa_list)} QA pairs to:\n - {out_json}\n - {out_jsonl}")


if __name__ == '__main__':
    main()
