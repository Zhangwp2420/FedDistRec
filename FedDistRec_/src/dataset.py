import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import random
from logger import DATA_ROOT
import os

class FederatedRecDataset:

    def __init__(self, dataset: str):

        self.inter_path = os.path.join(DATA_ROOT,dataset,'inter.csv')
        self.category_path = os.path.join(DATA_ROOT,dataset,'category.csv')
   
        self.user2id = {}
        self.item2id = {}
        self.cat2id = {}
        self.id2item = {}
        self.n_users = 0
        self.n_items = 0
        self.n_cats = 0

        # User-wise data: {user_id: [(item_id, ts), ...]}
        self.user_interactions = defaultdict(list)
    
        # Final split data: {user_id: {'train': [...], 'val': [...], 'test': [...]}}
        self.user_data_split = {}

        # Category distribution per user (on train set)
        self.user_category_dist = {}  # {user_id: {cat_id: count, ...}, ...}

        # Pre-sampled negative items for evaluation
        self.user_negatives_val = {}  # {user_id: [item_id1, item_id2, ...]}
        self.user_negatives_test = {}  # {user_id: [item_id1, item_id2, ...]}

        # Load and process
        self._load_data()
        self._split_data()
        self._compute_category_distribution()
        self._pre_sample_negatives()

     

    def _load_data(self):

        inter_df = pd.read_csv(self.inter_path, header=None, 
                            names=['userId', 'itemId', 'rating', 'timestamp'])
      
        cat_df = pd.read_csv(self.category_path, header=None, names=['itemId','categoryId'])

        all_users = sorted(inter_df['userId'].unique())
        all_items = sorted(inter_df['itemId'].unique())
        all_cats = sorted(cat_df['categoryId'].str.split('|').explode().str.strip().unique())
       
        self.user2id = {uid: idx for idx, uid in enumerate(all_users)}
        self.item2id = {iid: idx for idx, iid in enumerate(all_items)}
        self.cat2id = {cid: idx for idx, cid in enumerate(all_cats)}
        self.id2item = {idx: iid for iid, idx in self.item2id.items()}

        self.n_users = len(self.user2id)
        self.n_items = len(self.item2id)
        self.n_cats = len(self.cat2id)

        self.item_category_map = (
            cat_df.set_index('itemId')['categoryId']
            .apply(lambda x: x.split('|'))
            .to_dict()
        )
      
        self.item_category_id_map = {
            self.item2id[iid]: [self.cat2id[cat] for cat in cats if cat in self.cat2id]
            for iid, cats in self.item_category_map.items()
            if iid in self.item2id
        }
        grouped = inter_df.groupby('userId')
        for uid_str, group in grouped:
            uid = self.user2id[uid_str]
            records = []
            for _, row in group.iterrows():
                iid_str = row['itemId']
                if iid_str in self.item2id:
                    records.append((
                        self.item2id[iid_str],
                        row['timestamp'],
                     
                    ))
            records.sort(key=lambda x: x[1])
            self.user_interactions[uid] = records

   

    def _split_data(self):

        for uid, items_ts in self.user_interactions.items():

            num_items = len(items_ts)
            n_test = max(1, int(0.2 * num_items))
            n_val = max(1, int(0.1 * num_items))
            n_train = num_items - n_test - n_val
            split_idx = [n_train, n_val, n_test]

            train = items_ts[:split_idx[0]]
            val = items_ts[split_idx[0] : split_idx[0] + split_idx[1]]
            test = items_ts[split_idx[0] + split_idx[1]:]

            self.user_data_split[uid] = {
                'train': [(iid, 1.0) for iid, _ in train],
                'val': [(iid, 1.0) for iid, _  in val],
                'test': [(iid, 1.0) for iid, _ in test]
            }     

        self.user_interactions.clear()


    def _compute_category_distribution(self):

        for uid, data in self.user_data_split.items():

            train_items = [iid for iid, _ in data['train']]
            all_cat_ids = []
            for iid in train_items:
                if iid in self.item_category_id_map:
                    all_cat_ids.extend(self.item_category_id_map[iid])  
            
            if len(all_cat_ids) == 0:
                cat_vector = np.zeros(self.n_cats, dtype=np.float32)
            else:
                cat_vector = np.bincount(all_cat_ids, minlength=self.n_cats).astype(np.float32)
                cat_vector /= cat_vector.sum()  

            self.user_category_dist[uid] = cat_vector

    def get_user_category_vector(self, user_id: int) -> np.ndarray:

        return self.user_category_dist.get(user_id, np.zeros(self.n_cats))
   

    def _pre_sample_negatives(self, num_negatives=99, seed=2025):
      
        rng = np.random.Generator(np.random.PCG64(seed))
        
        for uid in range(self.n_users):
        
            non_interacted = self.get_all_negative_items(uid)
            assert len(non_interacted) >= 2 * num_negatives, \
                f"User {uid} has only {len(non_interacted)} non-interacted items, need at least {2 * num_negatives}"

            neg_val_test = rng.choice(non_interacted, size=num_negatives*2, replace=False)
            neg_val = neg_val_test[:num_negatives].tolist()      
            neg_test = neg_val_test[num_negatives:].tolist()    
            self.user_negatives_val[uid] = neg_val
            self.user_negatives_test[uid] = neg_test
    

    def get_evaluate_negatives(self, user_id, mode='val'):

        if mode == 'val':
            return self.user_negatives_val.get(user_id, [])
        elif mode == 'test':
            return self.user_negatives_test.get(user_id, [])
        else:
            raise ValueError("Mode must be 'val' or 'test'")

    def get_all_negative_items(self, user_id):

        interacted_items = set()
        for m in ['train', 'val', 'test']:
            for item, _ in self.user_data_split[user_id][m]:
                interacted_items.add(item)
        non_interacted = list(set(range(self.n_items)) - interacted_items)
        return non_interacted

    def get_train_negative_items(self, user_id):

        return list(set(self.get_all_negative_items(user_id)) \
            - set(self.user_negatives_val[user_id]) - set(self.user_negatives_test[user_id]))
       
    

class UserItemRatingDataset(torch.utils.data.Dataset):
 
    def __init__(self, items, ratings):
        self.items = items
        self.ratings = ratings
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx], self.ratings[idx]


if __name__ == "__main__":
    dataset = 'TG'
    FD = FederatedRecDataset(dataset)
    print(FD.top_k_sim_users[0])
    print(FD.get_user_category_vector(0))
    for i in FD.top_k_sim_users[0]:
        print(FD.get_user_category_vector(i))