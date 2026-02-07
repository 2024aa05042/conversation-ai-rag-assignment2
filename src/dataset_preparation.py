"""
Dataset Preparation for Hybrid RAG Assignment - TEST VERSION (10 docs).
SCALABILITY POINTS MARKED WITH ### SCALE HERE ###
Change to 500 for full assignment.
"""
import json
import os
import random
import re
from typing import List, Dict
import wikipediaapi
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import wikipedia
import ssl

# Fix SSL certificate issue for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

# Config - SCALABILITY POINTS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIXED_URLS_FILE = os.path.join(DATA_DIR, 'fixedurls.json')  ### CHANGE: _test for separate test files
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')
MIN_PAGE_WORDS = 800
CHUNK_TOKEN_MIN, CHUNK_TOKEN_MAX = 200, 400
OVERLAP_TOKENS = 50

### SCALE HERE #1: For testing=4 fixed, full=200 fixed
NUM_FIXED = 200
### SCALE HERE #2: For testing=6 random, full=300 random  
NUM_RANDOM = 5

def generate_fixed_urls(num: int = NUM_FIXED) -> List[str]:
    """Generate fixed URLs. ### Diverse sample for test; scales automatically."""
    fixed_titles = [
        'Quantum mechanics', 'World War II', 'India', 'Python (programming language)',  # Test diversity
        'Eiffel Tower', 'Machine learning', 'Ganges', 'Albert Einstein'
    ]
    fixed_urls = []
    for title in fixed_titles[:num]:  # Slice for num
        print("Title selected is" , title)
        try:
            page = wikipedia.page(title)
            if len(page.content.split()) >= MIN_PAGE_WORDS:
                fixed_urls.append(page.url)
        except:
            continue
    while len(fixed_urls) < num:  # Pad
        try:
            rand_title = wikipedia.random()
            page = wikipedia.page(rand_title)
            if len(page.content.split()) >= MIN_PAGE_WORDS:
                fixed_urls.append(page.url)
        except:
            continue
    with open(FIXED_URLS_FILE, 'w') as f:
        json.dump(fixed_urls, f, indent=2)
    print(f"✅ Saved {len(fixed_urls)} FIXED URLs: {FIXED_URLS_FILE}")
    return fixed_urls[:num]

def load_fixed_urls() -> List[str]:
    if not os.path.exists(FIXED_URLS_FILE):
        return generate_fixed_urls()
    with open(FIXED_URLS_FILE, 'r') as f:
        return json.load(f)

def fetch_wiki_text(url: str) -> str:
    """Fetch/clean text."""
    wiki = wikipediaapi.Wikipedia(user_agent='HybridRAGTest', language='en')
    title = url.split('/wiki/')[-1].replace('_', ' ')
    page = wiki.page(title)
    if not page.exists():
        return ""
    text = page.text
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = BeautifulSoup(text, 'html.parser').get_text()
    return text if len(text.split()) >= MIN_PAGE_WORDS else ""

def chunk_text(text: str, url: str, title: str) -> List[Dict]:
    """Chunk 200-400 tokens with 50-token overlap."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0
    for sent in sentences:
        sent_tokens = len(word_tokenize(sent))
        if current_tokens + sent_tokens > CHUNK_TOKEN_MAX and current_chunk:
            chunk = {
                'chunk_id': f"{title.replace(' ', '_')}_{chunk_id}",
                'text': ' '.join(current_chunk),
                'tokens': current_tokens,
                'url': url,
                'title': title
            }
            if CHUNK_TOKEN_MIN <= chunk['tokens'] <= CHUNK_TOKEN_MAX:
                chunks.append(chunk)
            # Overlap: Keep sentences from end until we have ~OVERLAP_TOKENS
            overlap_chunk = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                s_tokens = len(word_tokenize(s))
                if overlap_tokens + s_tokens <= OVERLAP_TOKENS:
                    overlap_chunk.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break
            current_chunk = overlap_chunk
            current_tokens = overlap_tokens
            chunk_id += 1
        current_chunk.append(sent)
        current_tokens += sent_tokens
    # Final
        # Final chunk flush: ensure last collected sentences are emitted if they meet size.
        if current_chunk:
        chunk = {
            'chunk_id': f"{title.replace(' ', '_')}_{chunk_id}",
            'text': ' '.join(current_chunk),
            'tokens': current_tokens,
            'url': url,
            'title': title
        }
        if CHUNK_TOKEN_MIN <= chunk['tokens'] <= CHUNK_TOKEN_MAX:
            chunks.append(chunk)
    return chunks

def prepare_dataset() -> None:
    """Main: 10 docs total."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    fixed_urls = load_fixed_urls()
    random_urls = []
    for _ in range(NUM_RANDOM * 2):  # Extra attempts for valid pages
        try:
            title = wikipedia.random()
            page = wikipedia.page(title)
            if len(page.content.split()) >= MIN_PAGE_WORDS:
                random_urls.append(page.url)
                if len(random_urls) == NUM_RANDOM:
                    break
        except:
            continue
    
    all_urls = fixed_urls + random_urls
    print(f"🚀 Processing {len(all_urls)} URLs ({NUM_FIXED} fixed + {NUM_RANDOM} random)")
    
    all_chunks = []
    for i, url in enumerate(all_urls, 1):
        print(f"📥 Fetching {i}/{len(all_urls)}: {url.split('/wiki/')[-1][:50]}...")
        title = url.split('/wiki/')[-1].replace('_', ' ')
        text = fetch_wiki_text(url)
        if text:
            chunks = chunk_text(text, url, title)
            all_chunks.extend(chunks)
            print(f"   → Added {len(chunks)} chunks ({sum(c['tokens'] for c in chunks)} tokens)")
    
    with open(CHUNKS_FILE, 'w') as f:
        json.dump(all_chunks, f, indent=1)  # indent=1 for smaller file
    
    print(f"\n🎉 SUCCESS: {len(all_chunks)} chunks → {CHUNKS_FILE}")
    print(f"   Avg tokens: {sum(c['tokens'] for c in all_chunks)/len(all_chunks):.0f}")
    print("\n### TO SCALE TO 500: Edit lines 33-34: NUM_FIXED=200, NUM_RANDOM=300")
    print("### Rename files back: fixedurls.json, chunks.json")

if __name__ == "__main__":
    prepare_dataset()
