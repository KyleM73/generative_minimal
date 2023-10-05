from typing import Any, Callable, List

import torch

class GAN(torch.nn.Module):
    def __init__(self, 
                 in_size: int, 
                 in_channels: int, 
                 latent_dim: int, 
                 hidden_dims: List = None, 
                 activation_func: Callable = torch.nn.LeakyReLU,
                 **kwargs
                 ) -> None:
        super(GAN, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.activation_func = activation_func
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]

        # generator
        generator_layers = self.block(self.latent_dim, hidden_dims[0], normalize=False)
        for i in range(len(hidden_dims)-1):
            generator_layers.extend(self.block(hidden_dims[i], hidden_dims[i+1]))
        generator_layers.append(torch.nn.Linear(hidden_dims[-1], in_channels * in_size**2))
        generator_layers.append(torch.nn.Tanh())

        self.generator = torch.nn.Sequential(*generator_layers)
        self.generator.apply(self.init_weights)

        # discriminator
        hidden_dims.reverse()
        discriminator_layers = [torch.nn.Linear(in_channels * in_size**2, hidden_dims[0])]
        for i in range(len(hidden_dims)-1):
            discriminator_layers.extend(self.block(hidden_dims[i], hidden_dims[i+1]))
        discriminator_layers.append(torch.nn.Linear(hidden_dims[-1], 1))
        discriminator_layers.append(torch.nn.Sigmoid())

        self.discriminator = torch.nn.Sequential(*discriminator_layers)
        self.discriminator.apply(self.init_weights)

    def init_weights(self, l: Any) -> None:
        if isinstance(l, torch.nn.Linear) or isinstance(l, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(l.weight)
            if isinstance(l, torch.nn.Linear): l.bias.data.fill_(0.01)

    def make_block(self, in_dim: int, out_dim: int, normalize: bool = True) -> List:
        layers = [torch.nn.Linear(in_dim, out_dim, bias=not normalize)]
        if normalize:
            layers.append(torch.nn.BatchNorm1d(out_dim))
        layers.append(self.activation_func())
        return layers
    
    def forward(self, input: torch.Tensor)
    
    def sample(self, batch_size, **kwargs) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim)
        return self.generate(z)

    def generate(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(input, **kwargs)[0]

    def loss(self, input: torch.Tensor, output: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kld_weight: float) -> dict:
        reconstruction_loss = torch.nn.functional.mse_loss(input, output)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        loss = reconstruction_loss + kld_weight * kld_loss
        return {"loss" : loss, "reconstruction" : reconstruction_loss.detach(), "kld" : kld_loss.detach()}

