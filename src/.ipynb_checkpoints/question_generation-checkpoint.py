"""
2.1 Question Generation - 100 Diverse QA Pairs
🔧 Edit NUM_QA_TOTAL = 100 on line 12
"""

import json
import random
import os
from transformers import pipeline

# 🔧 CONFIGURATION
NUM_QA_TOTAL = 25      # ← 100 for submission, 24 for testing
QA_PER_TYPE = 5       # 25×4 types = 100
CHUNKS_FILE = "data/chunks_test.json"
OUTPUT_FILE = f"data/qa_eval_{NUM_QA_TOTAL}.json"

def load_chunks():
    """Load your chunks_test.json"""
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"{CHUNKS_FILE} missing! Run dataset_preparation.py")
    
    with open(CHUNKS_FILE, 'r') as f:
        chunks = json.load(f)
    
    # Robust format handling
    processed = []
    for chunk in chunks:
        processed.append({
            'id': chunk.get('id', random.randint(1000,9999)),
            'text': chunk.get('text', '')[:800],
            'url': chunk.get('url', chunk.get('metadata', {}).get('url', 'unknown')),
            'title': chunk.get('title', chunk.get('metadata', {}).get('title', 'No title'))
        })
    return processed

def init_generator():
    """Fast local model"""
    print("🤖 Loading distilgpt2...")
    return pipeline("text-generation", model="distilgpt2", device=-1)

def generate_prompts():
    """4 Question types per assignment spec"""
    return {
        "factual": "From this text, what is: ",
        "comparative": "In this text, compare: ",
        "inferential": "Based on this text, why: ",
        "multihop": "From this text, how does: "
    }

def main():
    print(f"🎯 Generating {NUM_QA_TOTAL} QA pairs...")
    chunks = load_chunks()
    print(f"📦 {len(chunks)} chunks loaded")
    
    generator = init_generator()
    qa_pairs = []
    prompts = generate_prompts()
    types = list(prompts.keys())
    
    for qa_type in types:
        print(f"\n🧠 {qa_type.upper()} ({QA_PER_TYPE})...")
        type_chunks = random.sample(chunks, min(QA_PER_TYPE, len(chunks)))
        
        for chunk in type_chunks:
            prompt = prompts[qa_type] + chunk['text'][:500] + "?"
            result = generator(prompt, max_length=120, do_sample=True)[0]['generated_text']
            
            question = result.replace(prompt, "").strip()
            if not question.endswith('?'):
                question += "?"
            
            qa_pairs.append({
                "question": question[:200],
                "ground_truth": chunk['text'][:250] + "...",
                "source_url": chunk['url'],
                "chunk_id": chunk['id'],
                "doc_title": chunk['title'][:80],
                "qa_type": qa_type
            })
            print(f"  Q: {question[:60]}...")
    
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(qa_pairs[:NUM_QA_TOTAL], f, indent=1)
    
    print(f"\n✅ {len(qa_pairs)} QA saved → {OUTPUT_FILE}")
    print("Types:", {t: sum(1 for qa in qa_pairs if qa['qa_type']==t) for t in types})

if __name__ == "__main__":
    main()
