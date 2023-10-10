from typing import Any, Callable, List

import torch

class WGAN(torch.nn.Module):
    def __init__(self, 
                 in_size: int, 
                 in_channels: int, 
                 latent_dim: int, 
                 hidden_dims: List = None, 
                 activation_func: Callable = torch.nn.LeakyReLU,
                 **kwargs
                 ) -> None:
        super(WGAN, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.activation_func = activation_func
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        self.device = kwargs["device"]

        # generator
        generator_layers = self.make_block(self.latent_dim, hidden_dims[0], normalize=False)
        for i in range(len(hidden_dims)-1):
            generator_layers.extend(self.make_block(hidden_dims[i], hidden_dims[i+1]))
        generator_layers.append(torch.nn.Linear(hidden_dims[-1], in_channels * in_size**2, device=self.device))
        generator_layers.append(torch.nn.Tanh())
        generator_layers.append(torch.nn.Unflatten(1, (in_channels, in_size, in_size)))

        self.generator = torch.nn.Sequential(*generator_layers)
        self.generator.apply(self.init_weights)

        # discriminator
        hidden_dims.reverse()
        discriminator_layers = [torch.nn.Flatten()]
        discriminator_layers.append(torch.nn.Linear(in_channels * in_size**2, hidden_dims[0], device=self.device))
        for i in range(len(hidden_dims)-1):
            discriminator_layers.extend(self.make_block(hidden_dims[i], hidden_dims[i+1]))
        discriminator_layers.append(torch.nn.Linear(hidden_dims[-1], 1, device=self.device))
        discriminator_layers.append(torch.nn.Sigmoid())

        self.discriminator = torch.nn.Sequential(*discriminator_layers)
        self.discriminator.apply(self.init_weights)

    def init_weights(self, l: Any) -> None:
        if isinstance(l, torch.nn.Linear) or isinstance(l, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(l.weight)
            if isinstance(l, torch.nn.Linear) and l.bias is not None: l.bias.data.fill_(0.01)

    def make_block(self, in_dim: int, out_dim: int, normalize: bool = True) -> List:
        layers = [torch.nn.Linear(in_dim, out_dim, bias=not normalize, device=self.device)]
        if normalize:
            layers.append(torch.nn.BatchNorm1d(out_dim, device=self.device))
        layers.append(self.activation_func())
        return layers
    
    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        return self.sample(input.size(0))
    
    def sample(self, batch_size: int, **kwargs) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        return self.generate(z)

    def generate(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.generator(input)
    
    def discriminate(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.discriminator(input)
    
    def calculate_loss(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy(input, label)