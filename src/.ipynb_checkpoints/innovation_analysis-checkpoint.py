"""
Part 2.3: SIMPLIFIED Innovation (Plots + Analysis)
No complex temp objects - Uses your WORKING evaluator!
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from evaluation import RAGEvaluator

plt.style.use('default')
sns.set_palette("husl")

class SimpleInnovationAnalyzer:
    def __init__(self, evaluator):
        self.evaluator = evaluator
    
    def ablation_k_values(self, qa_pairs):
        """Ablation: Different top-K values"""
        print("\n🔬 Ablation: Top-K Comparison...")
        k_values = [3, 5, 10]
        mrr_scores = []
        
        for k in k_values:
            mrr = self.evaluator.compute_mrr_url(qa_pairs[:10], top_k=k)
            mrr_scores.append(mrr)
            print(f"  K={k}: MRR={mrr:.3f}")
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, mrr_scores, 'o-', linewidth=3, markersize=10)
        plt.title('Ablation: MRR vs Top-K')
        plt.xlabel('Top-K Chunks')
        plt.ylabel('MRR@URL')
        plt.grid(True, alpha=0.3)
        plt.savefig('data/ablation_k.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return dict(zip(k_values, mrr_scores))
    
    def qa_type_performance(self, qa_pairs):
        """Performance by question type"""
        print("\n📊 QA Type Breakdown...")
        type_groups = {}
        
        for t in ['factual', 'comparative', 'inferential', 'multihop']:
            type_qas = [qa for qa in qa_pairs if qa['qa_type'] == t]
            if type_qas:
                mrr = self.evaluator.compute_mrr_url(type_qas[:5])
                type_groups[t] = mrr
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        types = list(type_groups.keys())
        scores = [type_groups[t] for t in types]
        plt.bar(types, scores, alpha=0.8, edgecolor='black')
        plt.title('MRR by Question Type')
        plt.ylabel('MRR@URL')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        for i, v in enumerate(scores):
            plt.text(i, v+0.01, f'{v:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig('data/qtype_mrr.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return type_groups
    
    def failure_analysis(self, qa_pairs):
        """Top 10 hardest questions"""
        print("\n🚨 Hardest Questions (MRR=0)...")
        hard_qas = []
        
        for qa in qa_pairs:
            chunks = self.evaluator.generator.fuser.fuse(qa['question'], 5)
            urls = [self.evaluator.normalize_url(self.evaluator.safe_get_url(c)) 
                   for c in chunks]
            gt_url = self.evaluator.normalize_url(self.evaluator.safe_get_url(qa))
            
            if gt_url not in urls:
                hard_qas.append({
                    'question': qa['question'],
                    'type': qa['qa_type'],
                    'gt_url': gt_url,
                    'got_urls': urls[:2]
                })
        
        # Save
        with open('data/hardest_questions.json', 'w') as f:
            json.dump(hard_qas[:8], f, indent=2)
        
        print(f"Found {len(hard_qas)} failures")
        print("Types:", Counter(q['type'] for q in hard_qas))
        
        return hard_qas[:8]
    
    def run_analysis(self, qa_file="data/qa_eval_24.json"):
        """🎯 Simple pipeline - 3 plots + insights"""
        print("🎓 INNOVATION ANALYSIS v2")
        print("="*40)
        
        with open(qa_file) as f:
            qa_pairs = json.load(f)
        
        print(f"Analyzing {len(qa_pairs)} questions...")
        
        # Run analyses
        ablation = self.ablation_k_values(qa_pairs)
        qtypes = self.qa_type_performance(qa_pairs)
        failures = self.failure_analysis(qa_pairs)
        
        # Summary
        summary = {
            'ablation_k': ablation,
            'qa_types': qtypes,
            'failures_found': len(failures),
            'failure_types': dict(Counter(f['type'] for f in failures))
        }
        
        with open('data/innovation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n✅ COMPLETE! Files created:")
        print("  ablation_k.png")
        print("  qtype_mrr.png") 
        print("  hardest_questions.json")
        print("  innovation_summary.json")
        
        print("\n📈 INSIGHTS:")
        print(f"• Best K: {max(ablation, key=ablation.get)}")
        print(f"• Hardest type: {max(summary['failure_types'], key=summary['failure_types'].get) if summary['failures_found'] else 'None'}")

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    analyzer = SimpleInnovationAnalyzer(evaluator)
    analyzer.run_analysis()
