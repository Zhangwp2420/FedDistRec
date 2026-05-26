import torch.nn as nn
import torch
from abc import ABC, abstractmethod
from dataset import UserItemRatingDataset
from torch.utils.data import DataLoader

class BaseFedRecModel(nn.Module, ABC):
   
    def __init__(self, args, dataset, client_id):
        super().__init__()
        self.args = args
        self.dataset = dataset 
        self.client_id = client_id
        self.device = torch.device(args.device)
        self.negative_items_tensor = torch.tensor(dataset.get_train_negative_items(client_id), device=self.device)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def cal_loss(self, *args, **kwargs):
    
        pass

    @abstractmethod
    def predict(self, item_ids):
      
        pass
    
    @abstractmethod
    def get_shared_parameters(self):
    
        pass
    
    @abstractmethod
    def load_shared_parameters(self, shared_params):
      
        pass

    @abstractmethod
    def get_personalized_parameters(self) -> dict:
      
        pass
    
    @abstractmethod
    def load_personalized_parameters(self, personalized_params: dict):
     
        pass

    @abstractmethod
    def create_optimizer(self, lr: float):
       
        pass

    def get_train_loader(self, batch_size=None, shuffle=True):
     
        if batch_size is None:
            batch_size = self.args.batch_size
            

        train_data = self.dataset.user_data_split[self.client_id]['train']
        items = [item for item, _ in train_data]
        ratings = [rating for _, rating in train_data]
       
        dataset = UserItemRatingDataset(items, ratings)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


    def fed_train_single_batch(self, batch, optimizer):
      
        items, ratings = batch
        items = items.to(self.device)
        ratings = ratings.to(self.device)
        
        optimizer.zero_grad()
        loss = self.cal_loss(items)
        loss.backward()
        optimizer.step()
        
        return loss.item()
 

    def local_train(self, train_loader, local_epochs, optimizer):
     
        self.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                loss = self.fed_train_single_batch(batch, optimizer)
                epoch_loss += loss
                num_batches += 1
            total_loss += epoch_loss
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss