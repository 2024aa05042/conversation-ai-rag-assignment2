"""
Part 1.4 Response Generation.
- Uses Flan-T5-base LLM
- Prompts: query + top-5 RRF chunks
- Generates concise answers
"""
import json
import os
import time
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from rrf_fusion import RRFFuser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks_test.json')
MODEL_NAME = 'google/flan-t5-base'  # Assignment recommended[file:9]
MAX_CONTEXT_TOKENS = 512  # Limit
MAX_NEW_TOKENS = 100
TOP_N_CHUNKS = 5

class ResponseGenerator:
    def __init__(self):
        print("🚀 Loading Flan-T5-base...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.fuser = RRFFuser()
        print("✅ LLM ready!")
    
    def generate(self, query: str, top_n: int = TOP_N_CHUNKS) -> Dict:
        """Full pipeline: RRF → context → LLM answer."""
        start_time = time.time()
        
        # 1. Retrieve fused chunks
        fused_chunks = self.fuser.fuse(query, n_final=top_n)
        
        # 2. Build context (token-limited)
        context = "\n\n".join([c['title'] + ": " + c['full_text'] for c in fused_chunks])
        context_tokens = len(self.tokenizer(context)['input_ids'])
        
        # Trim if too long
        if context_tokens > MAX_CONTEXT_TOKENS:
            trimmed = fused_chunks[:-1]  # Drop lowest RRF
            context = "\n\n".join([c['title'] + ": " + c['full_text'] for c in trimmed])
        
        # 3. Prompt
        prompt = f"Answer based ONLY on the context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # 4. Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=MAX_CONTEXT_TOKENS, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("Answer:")[-1].strip()  # Extract
        
        elapsed = time.time() - start_time
        
        return {
            'query': query,
            'answer': answer,
            'context_sources': len(fused_chunks),
            'context_tokens': context_tokens,
            'response_time': f"{elapsed:.2f}s",
            'fused_chunks': fused_chunks,
            'full_prompt': prompt[:300] + "..."  # Preview
        }

def main():
    generator = ResponseGenerator()
    
    test_queries = [
        "What are the key principles of quantum mechanics?",
        "When did World War II start and end?",
        "What is Python programming language used for?"
    ]
    
    print("\n🤖 Full RAG Generation Tests:")
    for query in test_queries:
        result = generator.generate(query)
        print(f"\nQ: {result['query']}")
        print(f"A: {result['answer']}")
        print(f"⏱️  {result['response_time']} | 📄 {result['context_sources']} sources")
        print("-" * 80)

if __name__ == "__main__":
    main()
