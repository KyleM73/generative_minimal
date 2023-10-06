import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show() 