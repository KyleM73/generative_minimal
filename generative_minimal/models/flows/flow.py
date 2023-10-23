from typing import Any, Callable, List

import torch

class NormalizingFlow(torch.nn.Module):
    def __init__(self, 
                 in_size: int, 
                 in_channels: int, 
                 latent_dim: int, 
                 hidden_dims: List = None, 
                 activation_func: Callable = torch.nn.LeakyReLU,
                 **kwargs
                 ) -> None:
        super(NormalizingFlow, self).__init__()
        pass