from typing import Any, Callable, List

import torch

class GAN(torch.nn.Module):
    def __init__(self, 
                 in_size: int, 
                 in_channels: int, 
                 latent_dim: int, 
                 hidden_dims: List = None, 
                 activation_func: Callable = torch.nn.LeakyReLU,
                 label_smoothing: float = 0.1,
                 label_noise: float = 0.1,
                 **kwargs
                 ) -> None:
        super(GAN, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.activation_func = activation_func
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        self.label_smoothing = label_smoothing
        self.label_noise = label_noise

        # generator
        generator_layers = self.make_block(self.latent_dim, hidden_dims[0], normalize=False)
        for i in range(len(hidden_dims)-1):
            generator_layers.extend(self.make_block(hidden_dims[i], hidden_dims[i+1]))
        generator_layers.append(torch.nn.Linear(hidden_dims[-1], in_channels * in_size**2))
        generator_layers.append(torch.nn.Tanh())
        generator_layers.append(torch.nn.Unflatten(1, (in_size, in_size, in_channels)))

        self.generator = torch.nn.Sequential(*generator_layers)
        self.generator.apply(self.init_weights)

        # discriminator
        hidden_dims.reverse()
        discriminator_layers = [torch.nn.Flatten()]
        discriminator_layers.append(torch.nn.Linear(in_channels * in_size**2, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            discriminator_layers.extend(self.make_block(hidden_dims[i], hidden_dims[i+1]))
        discriminator_layers.append(torch.nn.Linear(hidden_dims[-1], 1))
        discriminator_layers.append(torch.nn.Sigmoid())

        self.discriminator = torch.nn.Sequential(*discriminator_layers)
        self.discriminator.apply(self.init_weights)

    def init_weights(self, l: Any) -> None:
        if isinstance(l, torch.nn.Linear) or isinstance(l, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(l.weight)
            if isinstance(l, torch.nn.Linear) and l.bias is not None: l.bias.data.fill_(0.01)

    def make_block(self, in_dim: int, out_dim: int, normalize: bool = True) -> List:
        layers = [torch.nn.Linear(in_dim, out_dim, bias=not normalize)]
        if normalize:
            layers.append(torch.nn.BatchNorm1d(out_dim))
        layers.append(self.activation_func())
        return layers
    
    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        generated_imgs = self.generator(input)
        predicted_labels_g = self.discriminator(generated_imgs)
        predicted_labels_d = self.discriminator(generated_imgs.detach())
        predicted_labels_r = self.discriminator(kwargs["data"])
        return [generated_imgs, predicted_labels_g, predicted_labels_d, predicted_labels_r]
    
    def sample(self, batch_size, **kwargs) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim)
        return self.generate(z)

    def generate(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.generator(input)

    def loss(self, predicted_labels_g: torch.Tensor, predicted_labels_d: torch.Tensor, predicted_labels_r: torch.Tensor) -> dict:
        generator_loss = torch.nn.functional.binary_cross_entropy(
            predicted_labels_g,
            torch.abs(self.label_smoothing * torch.randn(predicted_labels_g.size()) \
                      + torch.where(torch.rand(predicted_labels_g.size()) > 1 - self.label_noise, 1, 0))
        ) # fool the discriminator
        discriminator_loss_generated = torch.nn.functional.binary_cross_entropy(
            predicted_labels_d,
            torch.abs(self.label_smoothing * torch.randn(predicted_labels_d.size())  \
                      + torch.where(torch.rand(predicted_labels_d.size()) > 1 - self.label_noise, 0, 1))
        ) # identify generated images
        discriminator_loss_real = torch.nn.functional.binary_cross_entropy(
            predicted_labels_r,
            torch.abs(self.label_smoothing * torch.randn(predicted_labels_r.size())  \
                      + torch.where(torch.rand(predicted_labels_r.size()) > 1 - self.label_noise, 1, 0))
        ) # identify real images
        discriminator_loss = (discriminator_loss_generated + discriminator_loss_real) / 2
        return {"g_loss" : generator_loss, "d_loss" : discriminator_loss}