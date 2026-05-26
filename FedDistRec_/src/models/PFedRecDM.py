import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.basemodel import BaseFedRecModel

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, max_period=10000):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_period = max_period
        
    def forward(self, timesteps):
        half_dim = self.latent_dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period)) * 
            torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.latent_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class DiffusionApproximator(nn.Module):
   
    def __init__(self, latent_dim, num_cats):
        super().__init__() 
        self.timestep_embedder = TimestepEmbedder(latent_dim)
        self.fusing_layer_public = nn.Sequential(
            nn.Linear(latent_dim * 2 + num_cats , latent_dim),
            nn.ReLU(),   
        )
        self.fusing_layer_private = nn.Sequential(
            nn.Linear(latent_dim * 2 + num_cats, latent_dim),
            nn.ReLU(),
        )
        
        self.fusion_transform = nn.Linear(latent_dim, latent_dim, bias=False) 

    def forward(self, xt, t, cat_distribution):

        t_emb = self.timestep_embedder(t)
        combined = torch.cat([cat_distribution, xt, t_emb], dim=-1)
        x0_hat_public = self.fusing_layer_public(combined)
        x0_hat_private = self.fusing_layer_private(combined)
        x0_hat_public_transformed = self.fusion_transform(x0_hat_public)
        x0_hat = x0_hat_public_transformed + x0_hat_private
        return x0_hat

class PFedRecDM(BaseFedRecModel):  

    def __init__(self, args, dataset, client_id):

        super().__init__(args, dataset, client_id)

        self.latent_dim = args.latent_dim

        self.item_embeddings = torch.nn.Embedding(dataset.n_items, self.latent_dim)

        self.score_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 1),  
        )
        self.lr = args.lr
        self.wd = args.wd

        self.user_cat_hist = torch.tensor(
            dataset.get_user_category_vector(client_id), 
            dtype=torch.float32, device=self.device
        )
  
        self.diffusion_steps = args.diffusion_steps
        self._setup_diffusion_parameters()
      
        self.nmc = args.nmc
        self.num_cats = dataset.n_cats
        self.lambda_rec_loss = args.lambda_rec_loss  
        self.lambda_mse_loss = args.lambda_mse_loss 

        self.diffusion_approximator = DiffusionApproximator(
            self.latent_dim, 
            self.num_cats,
        )

        self._init_weights()
        self.to(self.device)

    def _init_weights(self):

        nn.init.normal_(self.item_embeddings.weight, std=0.01)
      
        for m in self.diffusion_approximator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

        for layer in self.score_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


    def _setup_diffusion_parameters(self):
       
        scale = self.args.scale / self.diffusion_steps
        beta_start = scale * self.args.beta_start + self.args.beta_base
        beta_end = scale * self.args.beta_end + self.args.beta_base
        
        if beta_end > 1:
            beta_end = 1 / self.diffusion_steps + self.args.beta_base
        
        betas = torch.linspace(beta_start, beta_end, self.diffusion_steps, device=self.device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat((torch.tensor([1.0], device=self.device), alphas_bar[:-1]))
        

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - alphas_bar))
      
        p_mu_c1 = betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar)
        p_mu_c2 = (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar)
        p_sqrt_var = torch.sqrt(betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))
        
        self.register_buffer('p_mu_c1', p_mu_c1)
        self.register_buffer('p_mu_c2', p_mu_c2)
        self.register_buffer('p_sqrt_var', p_sqrt_var)
        
        self.reverse_steps = list(range(self.diffusion_steps))[::-1]

    def predict(self, item_ids):
        
        user_embeddings = self._get_user_embeddings(
            self.user_cat_hist,
            x0=None,
            num_samples=self.nmc
        )

        candidate_embs = self.item_embeddings(item_ids) 

        m = candidate_embs.size(0)
        user_expand = user_embeddings.unsqueeze(1).expand(-1, m, -1)
      
        item_expand = candidate_embs.unsqueeze(0).expand(self.nmc, -1, -1)

        input_pairs = torch.cat([user_expand, item_expand], dim=-1)  

        scores = self.score_mlp(input_pairs).squeeze(-1) 

        final_scores = torch.mean(scores, dim=0) 
        return final_scores
  
    

    def cal_loss(self, pos_items):
        
        bs = pos_items.size(0)
    
        rand_idx = torch.randint(0, len(self.negative_items_tensor), 
                                (bs, self.args.num_negative), 
                                device=self.device)
        neg_items = self.negative_items_tensor[rand_idx] 
        
     
        pos_embs = self.item_embeddings(pos_items)  
        neg_embs = self.item_embeddings(neg_items)  
        
       
        x0_hat = self._get_user_embeddings(self.user_cat_hist, x0 = pos_embs) 
        
        all_items = torch.cat([pos_embs.unsqueeze(1), neg_embs], dim=1)  
        
        x0_hat_expanded = x0_hat.unsqueeze(1).expand(-1, all_items.size(1), -1)  
        all_scores = self.score_mlp(torch.cat([x0_hat_expanded, all_items], dim=-1)).squeeze(-1)  
        
  
        labels = torch.zeros(bs, 1 + self.args.num_negative, device=self.device) 
        labels[:, 0] = 1  
   
        rec_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            all_scores, 
            labels
        )

        mse = torch.mean((pos_embs - x0_hat) ** 2, dim=-1)  
        mse_loss = torch.mean(mse)

        total_loss = (
            rec_loss * self.lambda_rec_loss  +
            mse_loss * self.lambda_mse_loss
        )
        return total_loss


    def _get_user_embeddings(self, user_cat_hist, x0=None, num_samples=None):
       
        if x0 is not None:
            bs = x0.size(0)
        elif num_samples is not None:
            bs = num_samples
        else:
            bs = 1

        if user_cat_hist.dim() == 1:
            user_cat_hist = user_cat_hist.unsqueeze(0)

        user_cat_hist = user_cat_hist.expand(bs, -1).contiguous()
        
        if x0 is not None:
            t = torch.randint(0, self.diffusion_steps, (bs,), device=self.device)
            noise = torch.randn_like(x0)
            xt = self.sqrt_alphas_bar[t].unsqueeze(-1) * x0 + self.sqrt_one_minus_alphas_bar[t].unsqueeze(-1) * noise
            return self.diffusion_approximator(
                xt, t, user_cat_hist
            )

        noise_xt = torch.randn((bs, self.latent_dim), device=self.device)   
        for step in self.reverse_steps:
            t = torch.full((bs,), step, device=self.device)
            x0_hat = self.diffusion_approximator(
                    noise_xt, t, user_cat_hist
                )
            model_mean = self.p_mu_c1[t].unsqueeze(-1) * x0_hat + self.p_mu_c2[t].unsqueeze(-1) * noise_xt

            if step > 0:      
                    model_sqrt_var = self.p_sqrt_var[t].unsqueeze(-1)
                    noise = torch.randn_like(noise_xt)
                    noise_xt = model_mean + model_sqrt_var * noise
            else:
                    noise_xt = model_mean

        return noise_xt
   

    def fed_train_single_batch(self, batch, optimizer):
       
        optimizer_mlp, optimizer_item = optimizer
        items, ratings = batch
        items = items.to(self.device)
        ratings = ratings.to(self.device)
        
        optimizer_mlp.zero_grad()
        loss = self.cal_loss(items)
        loss.backward()
        optimizer_mlp.step()

        optimizer_item.zero_grad()
        loss = self.cal_loss(items)
        loss.backward()
        optimizer_item.step()
        
        return loss.item()


    def get_shared_parameters(self) -> dict:
        """upload"""

        shared_params = {}
        param_tensor = self.item_embeddings.weight.data.clone()
        shared_params['item_embeddings.weight'] = param_tensor
        shared_modules = ['fusing_layer_public']
        for module_name in shared_modules:
            module = getattr(self.diffusion_approximator, module_name, None)
            if module is not None:
                for param_name, param in module.named_parameters():
                    full_name = f'diffusion_approximator.{module_name}.{param_name}'
                    param_tensor = param.data.clone()
                    shared_params[full_name] = param_tensor

        for name, param in self.score_mlp.named_parameters():
            param_tensor = param.data.clone()
            shared_params[f'score_mlp.{name}'] = param_tensor

        return shared_params
       
 
    def load_shared_parameters(self, shared_params: dict):
      
        for param_name, param_value in shared_params.items():
            if param_name == 'item_embeddings.weight':
                self.item_embeddings.weight.data.copy_(param_value.to(self.device))
            elif param_name.startswith('diffusion_approximator.'):
               
                parts = param_name[len('diffusion_approximator.'):].split('.', 1)  
                if len(parts) < 2:
                    continue
                module_name, sub_param_name = parts
                if module_name in ['fusing_layer_public']:
                    target_module = getattr(self.diffusion_approximator, module_name, None)
                    if target_module is not None:
                        for name, param in target_module.named_parameters():
                            if name == sub_param_name:
                                param.data.copy_(param_value.to(self.device))
                                break
            elif param_name.startswith('score_mlp.'):
                    mlp_param_name = param_name[len('score_mlp.'):]
                    for name, param in self.score_mlp.named_parameters():
                        if name == mlp_param_name:
                            param.data.copy_(param_value.to(self.device))
                            break   


    def get_personalized_parameters(self) -> dict:
        
        personalized_params = {}

        personalized_params['item_embeddings.weight'] = self.item_embeddings.weight.data.clone()
        private_modules = ['fusing_layer_private','fusion_transform']
        for module_name in private_modules:
            module = getattr(self.diffusion_approximator, module_name, None)
            if module is not None:
                for param_name, param in module.named_parameters():
                    full_name = f'diffusion_approximator.{module_name}.{param_name}'
                    personalized_params[full_name] = param.data.clone()
     
        return personalized_params
       
    def load_personalized_parameters(self, personalized_params: dict):
     
        for param_name, param_value in personalized_params.items():
            if param_name.startswith('diffusion_approximator.'):
                parts = param_name[len('diffusion_approximator.'):].split('.', 1)
                if len(parts) < 2:
                    continue
                module_name, sub_param_name = parts
                if module_name in ['fusing_layer_private','fusion_transform']:
                    target_module = getattr(self.diffusion_approximator, module_name, None)
                    if target_module is not None:
                        for name, param in target_module.named_parameters():
                            if name == sub_param_name:
                                param.data.copy_(param_value.to(self.device))
                                break
            elif param_name == 'item_embeddings.weight':
                self.item_embeddings.weight.data.copy_(param_value.to(self.device))
        
        
    def create_optimizer(self):

        optimizer_mlp = torch.optim.AdamW([    
            {'params': self.score_mlp.parameters(),'lr':self.lr, 'weight_decay':self.wd},
            {'params': self.diffusion_approximator.parameters(),'lr':self.lr, 'weight_decay':self.wd}
        ])
        
        optimizer_item = torch.optim.AdamW(
            self.item_embeddings.parameters(), 
            lr=self.lr, weight_decay=self.wd
        )

        return optimizer_mlp, optimizer_item
    
