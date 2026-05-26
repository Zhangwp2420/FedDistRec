"""
    Some handy functions for pytroch model training ...
"""
import torch
import logging
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_class(model_name):
    try:
        module = __import__(f'models.{model_name}', fromlist=[model_name])
        model_class = getattr(module, model_name)
        return model_class
    except ImportError:
        raise ImportError(f"Failed to import model module 'models.{model_name}': {str(e)}")
    except AttributeError:
        raise AttributeError(f"Class '{model_name}' not found in module 'models.{model_name}': {str(e)}")
