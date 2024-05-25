from typing import Any, Callable, List

import torch

from generative_minimal import utils

class EBM(torch.nn.Module):
    def __init__(self, 
                 in_size: int, 
                 in_channels: int, 
                 noise_scale: float,
                 step_size: float,
                 alpha: float,
                 hidden_dims: List = None, 
                 activation_func: Callable = torch.nn.GELU,
                 **kwargs
                 ) -> None:
        super(EBM, self).__init__()

        self.in_channels = in_channels
        self.noise_scale = noise_scale
        self.step_size = step_size
        self.alpha = alpha
        if hidden_dims is None:
            hidden_dims = [4, 8, 16]
        self.activation_func = activation_func
        self.device = kwargs["device"]

        # encoder
        encoder_layers = []
        for h_dim in hidden_dims:
            encoder_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=1, padding=1, bias=False, device=self.device),
                    torch.nn.BatchNorm2d(h_dim, device=self.device),
                    self.activation_func()
                )
            )
            in_channels = h_dim
        encoder_layers.append(torch.nn.Flatten())

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.encoder.apply(self.init_weights)

        self.enc_size = utils.get_encoder_size(in_size, hidden_dims, kernel=3, stride=1, padding=1)

        self.energy = torch.nn.Linear(hidden_dims[-1] * self.enc_size**2, 1, device=self.device)
        self.energy.apply(self.init_weights)

    def init_weights(self, l: Any) -> None:
        if isinstance(l, torch.nn.Linear) or isinstance(l, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(l.weight)
            if isinstance(l, torch.nn.Linear): l.bias.data.fill_(0.01)

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        return self.energy(self.encoder(input))
    
    def sample(self, inputs_neg: torch.Tensor, n_steps: int) -> torch.Tensor:
        for _ in range(n_steps):
            noise = torch.randn_like(inputs_neg, device=self.device) * self.noise_scale
            energy_neg = self.forward(inputs_neg)
            grad = torch.autograd.grad(energy_neg.sum(), inputs_neg)[0]
            inputs_neg = inputs_neg - self.step_size * grad + noise
        return inputs_neg.detach()
    
    def loss(self, energy_pos: torch.Tensor, energy_neg: torch.Tensor) -> torch.Tensor:
        loss = (energy_pos.mean() - energy_neg.mean()) + self.alpha * ((energy_pos ** 2).mean() + (energy_neg ** 2).mean())
        return loss