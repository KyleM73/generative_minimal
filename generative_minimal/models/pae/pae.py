from typing import Any, Callable, List

import torch

from generative_minimal import utils

## Plan
# input [B,N,T] 
# latent [B,M,T]
# fft curve fitting
# phase_mode: "Learned", "Analytic"
# output [B,T,N]
# reconstruction loss

class PAE(torch.nn.Module):
    def __init__(self, 
                 in_size: int, 
                 in_channels: int, 
                 latent_dim: int, 
                 hidden_dims: List = None, 
                 activation_func: Callable = torch.nn.GELU,
                 phase_mode: str = "learned", # options = "learned" | "analytic"
                 **kwargs
                 ) -> None:
        super(PAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.activation_func = activation_func
        if hidden_dims is None:
            hidden_dims = [self.latent_dim]
        self.phase_mode = phase_mode
        self.device = kwargs.get("device", "cpu")
        
        # encoder
        encoder_layers = []
        for h_dim in hidden_dims:
            encoder_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels, out_channels=h_dim, kernel_size=3, stride=1, padding="same", bias=True, device=self.device),
                    torch.nn.BatchNorm1d(h_dim, device=self.device),
                    self.activation_func()
                )
            )
            in_channels = h_dim
        #encoder_layers.append(torch.nn.Flatten())

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.encoder.apply(self.init_weights)

        #self.enc_size = self.dec_size = utils.get_encoder_size(in_size, hidden_dims, kernel=3, stride=1, padding=1)

        # A sin( 2*pi*F*t + P) + B
        coefs = torch.fft.rfft(x)
        B  = coefs[0] / x.size()[-1]
        power = coefs * torch.conj(coefs) # see if normalized or not
        freq = torch.fft.freqs()
        F = torch.sum(freq*power) / torch.sum(power)
        A = torch.pow(2*torch.sum(power)/x.size()[-1], 0.5)
        if phase_mode == "learned":
            self.phase_encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.LazyLinear(2),
                torch.nn.Tanh()
                )
            self.phase_encoder.apply(self.init_weights)
        else:
            self.phase_encoder = self.analytic_phase()


        self.mu = torch.nn.Linear(hidden_dims[-1] * self.enc_size**2, latent_dim, device=self.device)
        self.mu.apply(self.init_weights)

        self.logvar = torch.nn.Linear(hidden_dims[-1] * self.enc_size**2, latent_dim, device=self.device)
        self.logvar.apply(self.init_weights)

        # decoder
        hidden_dims.reverse()
        decoder_layers = [
            torch.nn.Linear(latent_dim, hidden_dims[0] * self.dec_size**2, device=self.device),
            torch.nn.Unflatten(-1, (hidden_dims[0], self.dec_size, self.dec_size))
        ]
        for i in range(len(hidden_dims)-1):
            decoder_layers.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=1, padding=1, bias=False, device=self.device),
                    torch.nn.BatchNorm2d(hidden_dims[i+1], device=self.device),
                    self.activation_func()
                )
            )
        decoder_layers.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, padding=1, device=self.device),
                torch.nn.BatchNorm2d(hidden_dims[-1], device=self.device),
                self.activation_func(),
                torch.nn.Conv2d(hidden_dims[-1], self.in_channels, kernel_size=3, stride=1, padding=1, device=self.device),
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
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.device)
        return mu + std * eps

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), input, mu, logvar]
    
    def sample(self, batch_size, **kwargs) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        return self.decode(z)

    def generate(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(input, **kwargs)[0]

    def loss(self, input: torch.Tensor, output: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kld_weight: float) -> dict:
        reconstruction_loss = torch.nn.functional.mse_loss(input, output)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        loss = reconstruction_loss + kld_weight * kld_loss
        return {"loss" : loss, "reconstruction" : reconstruction_loss.detach(), "kld" : kld_loss.detach()}