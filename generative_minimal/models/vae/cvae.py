from typing import Any, Callable, List

import torch

class CVAE(torch.nn.Module):
    def __init__(self, 
                 in_size: int, 
                 in_channels: int, 
                 latent_dim: int,
                 context_dim: int, 
                 hidden_dims: List = None, 
                 activation_func: Callable = torch.nn.GELU
                 ) -> None:
        super(CVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.activation_func = activation_func
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        
        # encoder
        self.context_embedding = torch.nn.Sequential(
            torch.nn.Linear(context_dim, in_size**2),
            torch.nn.Unflatten(-1, (1, in_size, in_size))
            )
        self.context_embedding.apply(self.init_weights)

        self.input_embedding = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.input_embedding.apply(self.init_weights)

        in_channels += 1 # for context embedding
        encoder_layers = []
        for h_dim in hidden_dims:
            encoder_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.BatchNorm2d(h_dim),
                    self.activation_func()
                )
            )
            in_channels = h_dim
        encoder_layers.append(torch.nn.Flatten())

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.encoder.apply(self.init_weights)

        self.enc_size = self.dec_size = self.get_encoder_size(in_size, hidden_dims, kernel=3, stride=1, padding=1)

        self.mu = torch.nn.Linear(hidden_dims[-1] * self.enc_size**2, latent_dim)
        self.mu.apply(self.init_weights)

        self.logvar = torch.nn.Linear(hidden_dims[-1] * self.enc_size**2, latent_dim)
        self.logvar.apply(self.init_weights)

        # decoder
        hidden_dims.reverse()
        decoder_layers = [
            torch.nn.Linear(latent_dim + context_dim, hidden_dims[0] * self.dec_size**2),
            torch.nn.Unflatten(-1, (hidden_dims[0], self.dec_size, self.dec_size))
        ]
        for i in range(len(hidden_dims)-1):
            decoder_layers.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.BatchNorm2d(hidden_dims[i+1]),
                    self.activation_func()
                )
            )
        decoder_layers.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(hidden_dims[-1]),
                self.activation_func(),
                torch.nn.Conv2d(hidden_dims[-1], self.in_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        )

        self.decoder = torch.nn.Sequential(*decoder_layers)
        self.decoder.apply(self.init_weights)

    def init_weights(self, l: Any) -> None:
        if isinstance(l, torch.nn.Linear) or isinstance(l, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(l.weight)
            if isinstance(l, torch.nn.Linear): l.bias.data.fill_(0.01)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        latent = self.encoder(input)
        mu = self.mu(latent)
        logvar = self.logvar(latent)
        return [mu, logvar]

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        return self.decoder(input)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, input: torch.Tensor, **kwargs):
        y = kwargs["context"].float()
        embedded_y = self.context_embedding(y)
        embedded_input = self.input_embedding(input)
        input_conditioned = torch.cat((embedded_input, embedded_y), dim=1)
        mu, logvar = self.encode(input_conditioned)
        z = self.reparameterize(mu, logvar)
        z_conditioned = torch.cat((z, y), dim=1)
        return [self.decode(z_conditioned), input, mu, logvar]
    
    def sample(self, batch_size, **kwargs) -> torch.Tensor:
        y = kwargs['labels'].float()
        z = torch.randn(batch_size, self.latent_dim)
        z_conditioned = torch.cat((z, y), dim=1)
        return self.decode(z_conditioned)

    def generate(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(input, **kwargs)[0]

    def loss(self, input: torch.Tensor, output: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kld_weight: float) -> dict:
        reconstruction_loss = torch.nn.functional.mse_loss(input, output)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        loss = reconstruction_loss + kld_weight * kld_loss
        return {"loss" : loss, "reconstruction" : reconstruction_loss.detach(), "kld" : kld_loss.detach()}

    def get_encoder_size(self, in_size: int, hidden_dims: List[int], kernel: int, stride: int, padding: int) -> int:
        s = in_size
        for _ in hidden_dims:
            s = (s-kernel+2*padding)//stride + 1
        return s