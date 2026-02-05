"""
Part 2.2: RAG Evaluation (FIXED - Detailed JSON + CSV Ready)
✅ MRR@URL + ROUGE-L + Hit@5
✅ detailed_eval.json (per-question)
✅ Works with ANY chunk format!
"""

import json
import time
from collections import defaultdict
from rouge_score import rouge_scorer
from urllib.parse import urlparse
from response_generation import ResponseGenerator

class RAGEvaluator:
    """
    Production-ready RAG evaluator
    Exports detailed per-question results!
    """
    
    def __init__(self):
        """Initialize pipeline + metrics"""
        print("🔧 Loading RAG system...")
        self.generator = ResponseGenerator()
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def safe_get_url(self, obj):
        """🔑 Extract URL from ANY chunk format"""
        if isinstance(obj, dict):
            if 'metadata' in obj and 'url' in obj['metadata']:
                return obj['metadata']['url']
            if 'url' in obj:
                return obj['url']
            if 'source' in obj:
                return obj['source']
        return "unknown_url"
    
    def normalize_url(self, url):
        """🧹 Clean URL for matching"""
        if url == "unknown_url":
            return url
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
    
    def compute_mrr_url(self, qa_pairs, top_k=10):
        """2.2.1 MRR@URL (Mandatory)"""
        print("\n📈 MRR@URL...")
        mrr_scores = []
        
        for i, qa in enumerate(qa_pairs):
            print(f"  {i+1}/{len(qa_pairs)}: {qa['question'][:40]}...")
            retrieved_chunks = self.generator.fuser.fuse(qa['question'], top_k)
            urls = list(set(self.normalize_url(self.safe_get_url(c)) 
                          for c in retrieved_chunks))
            gt_url = self.normalize_url(self.safe_get_url(qa))
            
            rank = next((j+1 for j, url in enumerate(urls) if url == gt_url), float('inf'))
            mrr_score = 1/rank if rank != float('inf') else 0
            mrr_scores.append(mrr_score)
        
        return sum(mrr_scores) / len(mrr_scores)
    
    def compute_hit_rate_k(self, qa_pairs, k=5):
        """Hit Rate@K"""
        print(f"\n🎯 Hit Rate@{k}...")
        hits = sum(1 for qa in qa_pairs if 
                  self.normalize_url(self.safe_get_url(qa)) in 
                  [self.normalize_url(self.safe_get_url(c)) 
                   for c in self.generator.fuser.fuse(qa['question'], k)])
        return (hits / len(qa_pairs)) * 100
    
    def compute_rouge_l(self, qa_pairs, top_n=3):
        """ROUGE-L Answer Quality"""
        print("\n📝 ROUGE-L...")
        rouge_scores = []
        for qa in qa_pairs:
            result = self.generator.generate(qa['question'], top_n=top_n)
            score = self.rouge.score(qa['ground_truth'], result['answer'])
            rouge_scores.append(score['rougeL'].fmeasure)
        return sum(rouge_scores) / len(rouge_scores)
    
    def save_detailed_results(self, qa_pairs):
        """📋 PER-QUESTION BREAKDOWN (for CSV/PDF)"""
        print("\n💾 Saving detailed results...")
        detailed = []
        
        for i, qa in enumerate(qa_pairs):
            # Retrieve
            retrieved = self.generator.fuser.fuse(qa['question'], 5)
            urls = [self.normalize_url(self.safe_get_url(c)) for c in retrieved]
            gt_url = self.normalize_url(self.safe_get_url(qa))
            
            # Generate answer
            answer_result = self.generator.generate(qa['question'], top_n=3)
            
            detailed.append({
                'id': i+1,
                'question': qa['question'],
                'qa_type': qa.get('qa_type', 'unknown'),
                'ground_truth': qa['ground_truth'][:150] + "..." if len(qa['ground_truth']) > 150 else qa['ground_truth'],
                'generated_answer': answer_result['answer'][:150] + "...",
                'gt_url': gt_url,
                'retrieved_urls': urls,
                'hit_5': gt_url in urls,
                'retrieved_chunks': len(retrieved),
                'response_time': answer_result.get('time', 0)
            })
        
        # Save JSON + CSV
        with open('data/detailed_eval.json', 'w') as f:
            json.dump(detailed, f, indent=1)
        
        df = pd.DataFrame(detailed)
        df.to_csv('data/detailed_results.csv', index=False)
        
        print("✅ data/detailed_eval.json + detailed_results.csv")
        return detailed
    
    def evaluate_full(self, qa_file="data/qa_eval_24.json"):
        """🎯 COMPLETE EVALUATION + DETAILED EXPORT"""
        print("🔍 Loading questions...")
        with open(qa_file, 'r') as f:
            qa_pairs = json.load(f)
        
        print(f"📊 Evaluating {len(qa_pairs)} questions...")
        start = time.time()
        
        # All metrics
        mrr = self.compute_mrr_url(qa_pairs)
        hit_rate = self.compute_hit_rate_k(qa_pairs)
        rouge = self.compute_rouge_l(qa_pairs)
        
        # Detailed export
        detailed = self.save_detailed_results(qa_pairs)
        
        # Summary
        results = {
            "mrr_url": mrr,
            "hit_rate_5": hit_rate,
            "rouge_l": rouge,
            "total_questions": len(qa_pairs),
            "avg_time_per_q": (time.time() - start) / len(qa_pairs),
            "detailed_rows": len(detailed)
        }
        
        with open("data/eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n🎯 RESULTS:")
        print(f"MRR@URL:     {mrr:.3f}")
        print(f"Hit Rate@5:  {hit_rate:.1f}%")
        print(f"ROUGE-L:     {rouge:.3f}")
        print("✅ All files saved!")
        
        return results

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_full()
