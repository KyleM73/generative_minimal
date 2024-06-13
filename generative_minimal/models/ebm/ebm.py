from typing import Any, Callable, List

import torch

from generative_minimal import utils

class EBM(torch.nn.Module):
    def __init__(self, 
                 in_size: int = 28, 
                 in_channels: int = 1, 
                 noise_scale: float = 0.005,
                 grad_clip: float = 0.03,
                 step_size: float = 10,
                 alpha: float = 0.1,
                 hidden_dims: List = None, 
                 activation_func: Callable = torch.nn.SiLU,
                 **kwargs
                 ) -> None:
        super(EBM, self).__init__()

        self.in_channels = in_channels
        self.noise_scale = noise_scale
        self.grad_clip = grad_clip
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
    
    def generate_samples(self, sample: torch.Tensor, n_steps: int = 60, save_intermediate_samples: bool = False) -> torch.Tensor:
        train_status = self.training
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        sample.requires_grad = True

        gradients_status = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        noise = torch.randn_like(sample, device=self.device)
        if save_intermediate_samples: samples = []
        
        for _ in range(n_steps):
            noise.normal_(0, self.noise_scale)
            sample.data.add_(noise.data)
            sample.data.clamp_(min=-1.0, max=1.0)

            sample_energy = self.forward(sample)
            sample_energy.sum().backward()
            sample.grad.data.clamp_(-self.grad_clip, self.grad_clip)

            sample.data.add_(-self.step_size * sample.grad.data)
            sample.grad.detach_()
            sample.grad.zero_()
            sample.data.clamp_(min=-1.0, max=1.0)

            if save_intermediate_samples:
                samples.append(sample.clone().detach())
        
        for p in self.parameters():
            p.requires_grad = True
        self.train(train_status)
        torch.set_grad_enabled(gradients_status)

        if save_intermediate_samples:
            return torch.cat(samples, dim=0)
        else:
            return sample
    
    def loss(self, energy_data: torch.Tensor, energy_samples: torch.Tensor) -> List[torch.Tensor]:
        # note: energy is inverted i.e. net(x) = -E(x)
        # loss = contrastive divergence + regularization
        loss_cd = -energy_data.mean() + energy_samples.mean()
        loss_reg = self.alpha * (energy_data ** 2 + energy_samples ** 2).mean()
        loss = loss_cd + loss_reg
        return [loss, loss_cd, loss_reg]