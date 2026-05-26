import numpy as np
import torch
from collections import defaultdict
import pandas as pd
import os

class FederatedEvaluator:

    def __init__(self, dataset, k=10):
        self.dataset = dataset
        self.k = k

    def evaluate_user(self, model, user_id, mode='test'):
      
        data_split = self.dataset.user_data_split[user_id][mode]
        test_items = np.array([item for item, _ in data_split])
        candidate_items = list(test_items)
        device = next(model.parameters()).device
        neg_items = self.dataset.get_evaluate_negatives(user_id, mode)
        candidate_items.extend(neg_items)
        item_tensor = torch.LongTensor(candidate_items).to(device)

        model.eval()
        with torch.no_grad():
            scores = model.predict(item_tensor)

        _, top_indices = torch.topk(scores, min(self.k, len(scores)))
        top_items = item_tensor[top_indices].cpu().numpy()

        hr = self._hit_rate(top_items, test_items)
        ndcg = self._ndcg(top_items, test_items)
        recall = self._recall(top_items, test_items)
        precision = self._precision(top_items, test_items)

        return {
            f'hr@{self.k}': hr,
            f'ndcg@{self.k}': ndcg,
            f'recall@{self.k}': recall,
            f'precision@{self.k}': precision,
        }


    def _hit_rate(self, rec, rel):
        
        return 1.0 if set(rec) & set(rel) else 0.0

    def _recall(self, rec, rel):
       
        if len(rel) == 0:
            return 0.0
        hits = len(set(rec) & set(rel))
        return hits / len(rel)

    def _precision(self, rec, rel):
        
        if len(rec) == 0:   
            return 0.0
        hits = len(set(rec) & set(rel))
        return hits / len(rec)

    def _ndcg(self, rec, rel):
        
        if rec.size == 0 or rel.size == 0:
            return 0.0
        
        k = len(rec)
        rel_set = set(rel)
        
        relevance = np.array([1.0 if item in rel_set else 0.0 for item in rec])
        
        ranks = np.arange(1, k + 1)  
        discount = np.log2(ranks + 1)  

        dcg = np.sum(relevance / discount)
        
        num_relevant = min(len(rel), k)
        ideal_relevance = np.zeros(k)
        ideal_relevance[:num_relevant] = 1.0
        idcg = np.sum(ideal_relevance / discount)
        
        return dcg / idcg if idcg > 0 else 0.0
      
      
    def evaluate_all(self, model, users, mode='test'):
       
        metrics_list = []
  
        for user_id in users:
            user_metrics = self.evaluate_user(model[user_id], user_id, mode=mode)
            metrics_list.append(user_metrics)

        final_metrics = defaultdict(list)

        for metrics in metrics_list:
            for key, value in metrics.items():
                    final_metrics[key].append(value)

        final_result = {}
        for key, values in final_metrics.items():
            final_result[key] = np.mean(values)
        
        return final_result            
