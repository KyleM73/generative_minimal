# generative_minimal
a repository for minimal implementations of common generative models

each model class contains a `train.py` script to train the associated model

# Models:
- Generative Adversarial Networks (GANs)
    - Generative Adversarial Network (GAN)
    - Deep Convolutional Generative Adversarial Network (DCGAN)
    - Wasserstein Generative Adversarial Network (WGAN)
- Variational Auto Encoder (VAEs)
    - Variation Auto Encoder (VAE)
    - Conditional Variational Auto Encoder (CVAE)
- Normalizing Flows
    - Normalizing Flow
- Diffusion Models
    - Denoising Diffusion Probabalistic Model (DDPM)
- Transformers
    - Encoder-only
    - Decoder-only
    - Encoder-Decoder

# Setup
- `conda create -n generative -y python=3.11 && conda activate generative`
- `git clone git@github.com:KyleM73/generative_minimal.git`
- `cd generative_minimal`
- `pip install -e .`
- `cd generative_minimal`
- `python get_datasets.py`
- `python models/<model>/train.py`