from typing import Any, Callable, List

import torch

class DCGAN(torch.nn.Module):
    def __init__(self, 
                 in_size: int, 
                 in_channels: int, 
                 latent_dim: int, 
                 hidden_dims: List = None, 
                 activation_func: Callable = torch.nn.LeakyReLU,
                 label_smoothing: float = 0.0,
                 label_noise: float = 0.01,
                 **kwargs
                 ) -> None:
        super(DCGAN, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.activation_func = activation_func
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.label_smoothing = label_smoothing
        self.label_noise = label_noise
        self.device = kwargs["device"]

        # generator
        generator_layers = [torch.nn.Unflatten(1, (self.latent_dim, 1, 1))] #[1,1]
        generator_layers.extend(self.make_deconv_block(self.latent_dim, hidden_dims[0])) #[3,3]
        for i in range(len(hidden_dims)-1):
            generator_layers.extend(self.make_deconv_block(hidden_dims[i], hidden_dims[i+1])) #[7,7], [15,15]
        generator_layers.extend(self.make_deconv_block(hidden_dims[-1], self.in_channels, final=True)) #[28,28]

        self.generator = torch.nn.Sequential(*generator_layers)
        self.generator.apply(self.init_weights)

        # discriminator
        hidden_dims.reverse()
        discriminator_layers = [*self.make_conv_block(self.in_channels, hidden_dims[0])]
        for i in range(len(hidden_dims)-1):
            discriminator_layers.extend(self.make_conv_block(hidden_dims[i], hidden_dims[i+1]))
        discriminator_layers.extend(self.make_conv_block(hidden_dims[-1], 1, final=True))
        discriminator_layers.append(torch.nn.Flatten())

        self.discriminator = torch.nn.Sequential(*discriminator_layers)
        self.discriminator.apply(self.init_weights)

    def init_weights(self, l: Any) -> None:
        if isinstance(l, torch.nn.Linear) or isinstance(l, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(l.weight)
            if isinstance(l, torch.nn.Linear) and l.bias is not None: l.bias.data.fill_(0.01)

    def make_conv_block(self, in_dim: int, out_dim: int, normalize: bool = True, final: bool = False) -> List:
        layers = [torch.nn.Conv2d(in_dim, out_dim, 
                    kernel_size=3 if not final else 2, stride=2 if not final else 1,
                    padding=0, bias=not normalize, device=self.device)]
        if not final: 
            if normalize:
                layers.append(torch.nn.BatchNorm2d(out_dim, device=self.device))
            layers.append(self.activation_func())
        else: 
            layers.append(torch.nn.Sigmoid())
        return layers
    
    def make_deconv_block(self, in_dim: int, out_dim: int, normalize: bool = True, final: bool = False) -> List:
        layers = [torch.nn.ConvTranspose2d(in_dim, out_dim, 
                    kernel_size=3 if not final else 4, stride=2,
                    padding=0 if not final else 2, bias=not normalize, device=self.device)]
        if not final: 
            if normalize:
                layers.append(torch.nn.BatchNorm2d(out_dim, device=self.device))
            layers.append(self.activation_func())
        else: 
            layers.append(torch.nn.Tanh())
        return layers
    
    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        generated_imgs = self.sample(input.size(0))
        predicted_labels_g = self.discriminator(generated_imgs)
        predicted_labels_d = self.discriminator(generated_imgs.detach())
        predicted_labels_r = self.discriminator(input)
        return [generated_imgs, predicted_labels_g, predicted_labels_d, predicted_labels_r]
    
    def sample(self, batch_size: int, **kwargs) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        return self.generate(z)

    def generate(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.generator(input)

    def loss(self, predicted_labels_g: torch.Tensor, predicted_labels_d: torch.Tensor, predicted_labels_r: torch.Tensor) -> dict:
        generator_loss = torch.nn.functional.binary_cross_entropy(
            predicted_labels_g,
            torch.ones_like(predicted_labels_g, device=self.device)
        ) # fool the discriminator
        discriminator_loss_generated = torch.nn.functional.binary_cross_entropy(
            predicted_labels_d,
            torch.zeros_like(predicted_labels_d, device=self.device)
        ) # identify generated images
        discriminator_loss_real = torch.nn.functional.binary_cross_entropy(
            predicted_labels_r,
            torch.ones_like(predicted_labels_r, device=self.device)
        ) # identify real images
        discriminator_loss = (discriminator_loss_generated + discriminator_loss_real) / 2
        return {"g_loss" : generator_loss, "d_loss" : discriminator_loss}
    
    def loss_with_noise(self, predicted_labels_g: torch.Tensor, predicted_labels_d: torch.Tensor, predicted_labels_r: torch.Tensor) -> dict:
        generator_loss = torch.nn.functional.binary_cross_entropy(
            predicted_labels_g,
            torch.abs(self.label_smoothing * torch.randn(predicted_labels_g.size(), device=self.device) \
                      + torch.where(torch.rand(predicted_labels_g.size(), device=self.device) > 1 - self.label_noise, 1, 0))
        ) # fool the discriminator
        discriminator_loss_generated = torch.nn.functional.binary_cross_entropy(
            predicted_labels_d,
            torch.abs(self.label_smoothing * torch.randn(predicted_labels_d.size(), device=self.device)  \
                      + torch.where(torch.rand(predicted_labels_d.size(), device=self.device) > 1 - self.label_noise, 0, 1))
        ) # identify generated images
        discriminator_loss_real = torch.nn.functional.binary_cross_entropy(
            predicted_labels_r,
            torch.abs(self.label_smoothing * torch.randn(predicted_labels_r.size(), device=self.device)  \
                      + torch.where(torch.rand(predicted_labels_r.size(), device=self.device) > 1 - self.label_noise, 1, 0))
        ) # identify real images
        discriminator_loss = (discriminator_loss_generated + discriminator_loss_real) / 2
        return {"g_loss" : generator_loss, "d_loss" : discriminator_loss}