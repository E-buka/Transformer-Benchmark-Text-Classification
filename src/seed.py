import numpy as np
import torch
from numpy import random 
import config 

def set_seed():
    torch.manual_seed(config.SEED) 
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    
