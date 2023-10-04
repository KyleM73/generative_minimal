import torch
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # mps is almost always slower
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
import numpy as np

from generative_minimal.models import VAE

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    # params
    batch_size = 1000
    latent_dim = 8
    epochs = 10

    # CIFAR10
    #transform = torchvision.transforms.Compose([
    #    torchvision.transforms.ToTensor(), # from PIL.Image.Image to torch.Tensor
    #    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # from [0,1] to [-1,1]
    #    ]) 
    #trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    #testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    #classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    #in_channels = 3

    # MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # from PIL.Image.Image to torch.Tensor
        torchvision.transforms.Normalize((0.5), (0.5)), # from [0,1] to [-1,1]
        ]) 
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform) #60k
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=transform) #10k
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    in_channels = 1

    # make dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, pin_memory=True)
    
    # define network
    net = VAE(in_size=28, in_channels=in_channels, latent_dim=latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    print(net)
    print()

    # train network
    for epoch in range(epochs):
        net.train()
        running_loss, running_recons, running_kld = 0, 0, 0
        for i, data in enumerate(trainloader, start=0):
            inputs, labels = data
            for param in net.parameters():
                param.grad = None
            generated, src, mu, logvar = net(inputs.to(DEVICE))
            loss_dict = net.loss(src, generated, mu, logvar, 1/len(trainloader))
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
            generated, src, mu, logvar = net(inputs.to(DEVICE))
            loss_dict = net.loss(src, generated, mu, logvar, 1/len(testloader))

            running_loss += loss_dict["loss"].detach()
            running_recons += loss_dict["reconstruction"]
            running_kld += loss_dict["kld"]
        print(
            "[{epoch}, test] loss: {loss} reconstruction loss: {recons} kld loss: {kld}"
            .format(epoch=epoch,loss=running_loss/(i+1),recons=running_recons/(i+1),kld=running_kld/(i+1))
        )
        print()
    
    imgs = net.cpu().sample(batch_size=64)
    imshow(torchvision.utils.make_grid(imgs))