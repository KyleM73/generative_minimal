import torch
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # mps is almost always slower
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
import numpy as np

from generative_minimal.models import VAE, CVAE

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    # params
    batch_size = 1000

    # CIFAR10
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # from PIL.Image.Image to torch.Tensor
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # from [0,1] to [-1,1]
        ]) 
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform) # 50k
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform) # 10k
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    in_channels = 3
    in_size = 32
    latent_dim = 8
    epochs = 10
    """

    # MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # from PIL.Image.Image to torch.Tensor
        torchvision.transforms.Normalize((0.5), (0.5)), # from [0,1] to [-1,1]
        ]) 
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform) # 60k
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=transform) # 10k
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    in_channels = 1
    in_size = 28
    latent_dim = 16
    epochs = 10

    # make dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, pin_memory=True)
    
    kld_weight_train = 1/len(trainloader)
    kld_weight_test = 1/len(testloader)
    
    # define network
    net = CVAE(in_size=in_size, in_channels=in_channels, latent_dim=latent_dim, context_dim=len(classes)).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    print(net)
    print()

    # train network
    for epoch in range(epochs):
        net.train()
        running_loss, running_recons, running_kld = 0, 0, 0
        for i, data in enumerate(trainloader, start=0):
            inputs, labels = data
            one_hot_labels = torch.nn.functional.one_hot(labels)
            for param in net.parameters():
                param.grad = None
            generated, src, mu, logvar = net(inputs.to(DEVICE), context=one_hot_labels.to(DEVICE))
            loss_dict = net.loss(src, generated, mu, logvar, kld_weight_train)
            loss_dict["loss"].backward()
            optimizer.step()

            running_loss += loss_dict["loss"].detach()
            running_recons += loss_dict["reconstruction"]
            running_kld += loss_dict["kld"]

            print(
                "[{epoch}, {batch}] loss: {loss} reconstruction loss: {recons} kld loss: {kld}"
                .format(epoch=epoch,batch=i,loss=loss_dict["loss"].detach(),recons=loss_dict["reconstruction"],kld=loss_dict["kld"])
                )
        print(
            "[{epoch}, train] loss: {loss} reconstruction loss: {recons} kld loss: {kld}"
            .format(epoch=epoch,loss=running_loss/(i+1),recons=running_recons/(i+1),kld=running_kld/(i+1))
        )
        net.eval()
        running_loss, running_recons, running_kld = 0, 0, 0
        for i, data in enumerate(testloader, start=0):
            inputs, labels = data
            one_hot_labels = torch.nn.functional.one_hot(labels)
            generated, src, mu, logvar = net(inputs.to(DEVICE), context=one_hot_labels.to(DEVICE))
            loss_dict = net.loss(src, generated, mu, logvar, kld_weight_test)

            running_loss += loss_dict["loss"].detach()
            running_recons += loss_dict["reconstruction"]
            running_kld += loss_dict["kld"]
        print(
            "[{epoch}, test] loss: {loss} reconstruction loss: {recons} kld loss: {kld}"
            .format(epoch=epoch,loss=running_loss/(i+1),recons=running_recons/(i+1),kld=running_kld/(i+1))
        )
        print()

    labels = torch.tensor([[i for _ in range(len(classes))] for i in range(len(classes))]).view(-1)
    imgs = net.cpu().sample(batch_size=100, context=torch.nn.functional.one_hot(labels))
    imshow(torchvision.utils.make_grid(imgs, nrow=10))