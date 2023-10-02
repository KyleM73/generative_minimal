import torchvision

# CIFAR10
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

# MNIST
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True)
