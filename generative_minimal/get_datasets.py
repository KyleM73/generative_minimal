import torchvision

from generative_minimal import ROOT_DIR

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), # from PIL.Image.Image to torch.Tensor
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # from [0,1] to [-1,1]

# CIFAR10
trainset = torchvision.datasets.CIFAR10(root="{}/data".format(ROOT_DIR), train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="{}/data".format(ROOT_DIR), train=False, download=True, transform=transform)

# MNIST
trainset = torchvision.datasets.MNIST(root="{}/data".format(ROOT_DIR), train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root="{}/data".format(ROOT_DIR), train=False, download=True, transform=transform)
