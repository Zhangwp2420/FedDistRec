import torch
import random
import copy
import numpy as np
from metrics import FederatedEvaluator
import datetime
import math
import os

class Engine(object):

    def __init__(self, config, dataset, model_class):   

        self.config = config  
        self.dataset = dataset   
        self.model_class = model_class   
        self.server_shared_params = None 
        self.client_personalized_params = {}
        self.evaluator = FederatedEvaluator(dataset, config.top_k)
        self._init_server_shared_params()

    def _init_server_shared_params(self):
     
        temp_model = self.model_class(self.config, self.dataset, client_id=0)
        shared_params = temp_model.get_shared_parameters()
    
        self.server_shared_params = {
            key: value.clone() for key, value in shared_params.items()
        }

        del temp_model
       

    def create_client_model(self, user_id):
        
        model = self.model_class(self.config, self.dataset, client_id=user_id)
        model.load_shared_parameters(self.server_shared_params)
        if user_id in self.client_personalized_params:
                model.load_personalized_parameters(
                    self.client_personalized_params[user_id]
        )

        return model
    

    def aggregate_shared_params(self, participant_params):

            aggregated_params = {}
            participant_count = len(participant_params)
            first_user = next(iter(participant_params.keys()))
            for param_name, param_value in participant_params[first_user].items():
                aggregated_params[param_name] = torch.zeros_like(param_value)
            
            for user_id, params in participant_params.items():
                for param_name, param_value in params.items():
                    aggregated_params[param_name] += param_value
        
            for param_name in aggregated_params:
                aggregated_params[param_name] /= participant_count

            self.server_shared_params = aggregated_params

    def fed_train_a_round(self, round_id):

        num_participants = int(self.dataset.n_users * self.config.clients_sample_ratio)
        participants = random.sample(range(self.dataset.n_users), num_participants)
     
        participant_shared_params = {} 
        all_loss = {}
        
        for user_id in participants:
           
            model = self.create_client_model(user_id)

            optimizer = model.create_optimizer()

            train_loader = model.get_train_loader(
                batch_size=self.config.batch_size,
                shuffle=True
            )
            avg_loss = model.local_train(
                train_loader, 
                self.config.local_epoch, 
                optimizer
            )

            self.client_personalized_params[user_id] = model.get_personalized_parameters()
            participant_shared_params[user_id] = model.get_shared_parameters()
            
            all_loss[user_id] = avg_loss

            del model
            torch.cuda.empty_cache()  
        
        self.aggregate_shared_params(participant_shared_params)

        return all_loss
    

    def evaluate(self, sample_users=None, mode='test'):
        
        if sample_users is None:
            sample_users = list(range(self.dataset.n_users))   

        models = {}

        for user_id in sample_users:
            models[user_id] = self.create_client_model(user_id)

        return self.evaluator.evaluate_all(models, sample_users, mode)
    

    def run(self, num_rounds):
      
        metrics = [
                'train_loss', 
                f'hr@{self.config.top_k}', 
                f'ndcg@{self.config.top_k}', 
                f'recall@{self.config.top_k}', 
                f'precision@{self.config.top_k}'   
            ]
        
        results = {metric: [] for metric in metrics}
        best_ndcg = 0.0
        best_round = -1
        rounds_without_improvement = 0
        best_server_params = None
        best_client_params = None
        
        for round_id in range(num_rounds):
           
            train_loss = self.fed_train_a_round(round_id)
            avg_train_loss = np.mean(list(train_loss.values()))
            results['train_loss'].append(avg_train_loss)

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # warm up - no evaluate
            if round_id < self.config.warmup:
                print(f"Round {round_id}/{num_rounds} [Warmup]: Train Loss = {avg_train_loss:.4f}")
                continue
                
            
            if round_id % self.config.eval_interval == 0 or round_id == num_rounds - 1:
                # val set
                eval_results = self.evaluate(mode='val')
                
                for metric in metrics[1:]:  # jump train_loss
                    results[metric].append(eval_results[metric])
                
                current_ndcg = eval_results[f'ndcg@{self.config.top_k}']  # modify

                print(f"[{current_time}] Round {round_id}/{num_rounds}: "
                    f"Train Loss = {avg_train_loss:.4f}, "
                    f"HR@{self.config.top_k} = {eval_results[f'hr@{self.config.top_k}']:.4f}, "
                    f"NDCG@{self.config.top_k} = {current_ndcg:.4f}, "
                    f"Recall@{self.config.top_k} = {eval_results[f'recall@{self.config.top_k}']:.4f}, "
                    f"Precision@{self.config.top_k} = {eval_results[f'precision@{self.config.top_k}']:.4f}")
                

                if current_ndcg > best_ndcg:
                    best_ndcg = current_ndcg
                    best_round = round_id
                    rounds_without_improvement = 0
                    
                    best_server_params = copy.deepcopy(self.server_shared_params)
                    best_client_params = copy.deepcopy(self.client_personalized_params)

                else:
                    rounds_without_improvement += 1
                    
                if rounds_without_improvement >= self.config.patience:
                    print(f"Early stopping at round {round_id}")
                    break


        if best_round >= 0:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] Loading best model from round {best_round}")

            self.server_shared_params = best_server_params
            self.client_personalized_params = best_client_params
            test_results = self.evaluate(mode='test')
            
            for metric in metrics[1:]:
                results[metric + '_test'] = test_results[metric]
            
            print("\nFinal Test Results:")
            print(f"HR@{self.config.top_k} = {test_results[f'hr@{self.config.top_k}']:.4f}")
            print(f"NDCG@{self.config.top_k} = {test_results[f'ndcg@{self.config.top_k}']:.4f}")
            print(f"Recall@{self.config.top_k} = {test_results[f'recall@{self.config.top_k}']:.4f}")
            print(f"Precision@{self.config.top_k} = {test_results[f'precision@{self.config.top_k}']:.4f}")
        
        return results
    
  