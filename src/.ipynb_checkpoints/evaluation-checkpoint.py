"""
Part 2.2: RAG Evaluation Framework (10 Marks)
✅ 2.2.1 MRR@URL (Mandatory 2 marks)
✅ 2.2.2 ROUGE-L + Hit Rate@5 (Custom 4 marks)  
🚀 Robust: Handles ANY chunk format!
"""

import json
import time
from collections import defaultdict
from rouge_score import rouge_scorer
from urllib.parse import urlparse
from response_generation import ResponseGenerator
from dense_retrieval import DenseRetriever
from sparse_retrieval import SparseRetriever
from rrf_fusion import RRFFuser

class RAGEvaluator:
    """
    Universal RAG Evaluator - Works with your chunk formats!
    Metrics: MRR@URL, ROUGE-L, Hit Rate@K
    """
    
    def __init__(self):
        """Initialize RAG pipeline + metrics"""
        print("🔧 Initializing RAG pipeline...")
        self.generator = ResponseGenerator()
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def safe_get_url(self, obj):
        """
        🔑 ROBUST URL Extraction (handles ALL chunk formats)
        Priority: metadata.url → url → source → "unknown"
        """
        if isinstance(obj, dict):
            # Chunk format 1: {"metadata": {"url": "..."}}
            if 'metadata' in obj and 'url' in obj['metadata']:
                return obj['metadata']['url']
            # Chunk format 2: {"url": "..."}
            if 'url' in obj:
                return obj['url']
            # Chunk format 3: {"source": "..."}
            if 'source' in obj:
                return obj['source']
        return "unknown_url"
    
    def normalize_url(self, url):
        """🧹 Clean URL for matching (remove trailing /)"""
        if url == "unknown_url":
            return url
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
    
    def compute_mrr_url(self, qa_pairs, top_k=10):
        """
        2.2.1 MRR @ URL Level (MANDATORY - 2 Marks)
        Formula: avg(1/rank) where rank = position of FIRST correct URL
        """
        print("\n📈 Computing MRR@URL...")
        mrr_scores = []
        
        for i, qa in enumerate(qa_pairs):
            print(f"  {i+1}/{len(qa_pairs)}: {qa['question'][:40]}...")
            
            # Retrieve top-K chunks
            retrieved_chunks = self.generator.fuser.fuse(qa['question'], top_k)
            
            # Extract UNIQUE URLs (deduplicated)
            urls = list(set(self.normalize_url(self.safe_get_url(c)) 
                          for c in retrieved_chunks))
            
            gt_url = self.normalize_url(self.safe_get_url(qa))
            
            # Find FIRST occurrence
            rank = next((j+1 for j, url in enumerate(urls) if url == gt_url), 
                       float('inf'))
            mrr_score = 1/rank if rank != float('inf') else 0
            mrr_scores.append(mrr_score)
        
        mrr_final = sum(mrr_scores) / len(mrr_scores)
        print(f"✅ MRR@URL = {mrr_final:.3f}")
        return mrr_final
    
    def compute_hit_rate_k(self, qa_pairs, k=5):
        """
        Hit Rate@K: % questions where correct URL appears in top-K
        High value = reliable retrieval!
        """
        print(f"\n🎯 Hit Rate@{k}...")
        hits = 0
        
        for qa in qa_pairs:
            retrieved_chunks = self.generator.fuser.fuse(qa['question'], k)
            retrieved_urls = [self.normalize_url(self.safe_get_url(c)) 
                            for c in retrieved_chunks]
            gt_url = self.normalize_url(self.safe_get_url(qa))
            
            if gt_url in retrieved_urls:
                hits += 1
        
        hit_rate = (hits / len(qa_pairs)) * 100
        print(f"✅ Hit@{k} = {hit_rate:.1f}%")
        return hit_rate
    
    def compute_rouge_l(self, qa_pairs, top_n=3):
        """
        ROUGE-L F1: Semantic overlap between generated vs ground truth answers
        High = generated answers contain similar content
        """
        print("\n📝 ROUGE-L (Answer Quality)...")
        rouge_scores = []
        
        for i, qa in enumerate(qa_pairs):
            if i % 5 == 0:
                print(f"  {i+1}/{len(qa_pairs)}")
            
            # Generate answer
            result = self.generator.generate(qa['question'], top_n=top_n)
            generated_answer = result['answer']
            ground_truth = qa['ground_truth']
            
            score = self.rouge.score(ground_truth, generated_answer)
            rouge_scores.append(score['rougeL'].fmeasure)
        
        rouge_avg = sum(rouge_scores) / len(rouge_scores)
        print(f"✅ ROUGE-L = {rouge_avg:.3f}")
        return rouge_avg
    
    def save_detailed_results(self, qa_pairs, metrics):
        """📊 Save per-question breakdown for analysis"""
        detailed = []
        for i, qa in enumerate(qa_pairs):
            retrieved = self.generator.fuser.fuse(qa['question'], 5)
            urls = [self.normalize_url(self.safe_get_url(c)) for c in retrieved]
            
            detailed.append({
                'id': i+1,
                'question': qa['question'],
                'ground_truth': qa['ground_truth'][:100],
                'retrieved_urls': urls,
                'mrr_url': metrics['mrr_url_per_q'][i] if 'mrr_url_per_q' in metrics else 'N/A'
            })
        
        with open('data/detailed_eval.json', 'w') as f:
            json.dump(detailed, f, indent=1)
    
    def evaluate_full(self, qa_file="data/qa_eval_24.json"):
        """
        🚀 MAIN PIPELINE: Single command evaluation!
        Saves: eval_results.json + detailed_eval.json
        """
        print("🔍 Loading QA pairs...")
        with open(qa_file, 'r') as f:
            qa_pairs = json.load(f)
        
        print(f"📊 {len(qa_pairs)} questions | {len(set(qa['source_url'] for qa in qa_pairs))} unique URLs")
        
        start_time = time.time()
        
        # Compute all metrics
        mrr = self.compute_mrr_url(qa_pairs)
        hit_rate = self.compute_hit_rate_k(qa_pairs)
        rouge = self.compute_rouge_l(qa_pairs)
        
        total_time = time.time() - start_time
        
        # Results dict
        results = {
            "mrr_url": mrr,
            "hit_rate_5": hit_rate,
            "rouge_l": rouge,
            "total_questions": len(qa_pairs),
            "avg_time_per_q": total_time / len(qa_pairs),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save summary
        with open("data/eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*50)
        print("🎉 EVALUATION COMPLETE!")
        print("="*50)
        print(f"MRR@URL:      {mrr:.3f}     📈 [0.0-1.0]")
        print(f"Hit Rate@5:   {hit_rate:.1f}%   🎯 [0-100%]")
        print(f"ROUGE-L:      {rouge:.3f}     📝 [0.0-1.0]") 
        print(f"Questions:    {len(qa_pairs)}")
        print(f"Time:         {total_time:.1f}s")
        print(f"Saved: data/eval_results.json")
        
        return results

if __name__ == "__main__":
    # 🔥 RUN EVALUATION
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_full("data/qa_eval_24.json")
