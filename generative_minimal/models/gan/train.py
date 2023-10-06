import torch
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # mps is almost always slower
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True # enables cuDNN auto-tuner

import matplotlib.pyplot as plt
import numpy as np

from generative_minimal.models import GAN
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    # CIFAR10
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # from PIL.Image.Image to torch.Tensor
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # from [0,1] to [-1,1]
        ]) 
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform) # 50k
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform) # 10k
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    n_classes = len(classes)
    in_channels = 3
    in_size = 32
    latent_dim = 128
    epochs = 10
    batch_size = 200
    """

    # MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # from PIL.Image.Image to torch.Tensor
        torchvision.transforms.Normalize((0.5), (0.5)), # from [0,1] to [-1,1]
        ]) 
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform) # 60k
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=transform) # 10k
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    n_classes = len(classes)
    in_channels = 1
    in_size = 28
    latent_dim = 32
    epochs = 10
    batch_size = 200

    # make dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, pin_memory=True)
    
    kld_weight_train = 1/len(trainloader)
    kld_weight_test = 1/len(testloader)
    
    # define network
    net = GAN(in_size=in_size, in_channels=in_channels, latent_dim=latent_dim, context_dim=n_classes).to(DEVICE)
    optimizer_G = torch.optim.Adam(net.generator.parameters(), lr=3e-4)
    optimizer_D = torch.optim.Adam(net.discriminator.parameters(), lr=3e-4)

    print(net)
    print()

    # train network
    for epoch in range(epochs):
        net.train()
        running_loss_G, running_loss_D = 0, 0
        for i, data in enumerate(trainloader, start=0):
            inputs, labels = data
            one_hot_labels = torch.nn.functional.one_hot(labels)
            for param in net.parameters():
                param.grad = None
            z = torch.randn((batch_size, latent_dim))
            generated_imgs, predicted_labels_g, predicted_labels_d, predicted_labels_r = net(z, data=inputs.to(DEVICE), context=one_hot_labels.to(DEVICE))
            loss_dict = net.loss(predicted_labels_g, predicted_labels_d, predicted_labels_r)
            loss_dict["g_loss"].backward()
            optimizer_G.step()

            loss_dict["d_loss"].backward()
            optimizer_D.step()

            running_loss_G += loss_dict["g_loss"].detach()
            running_loss_D += loss_dict["d_loss"].detach()

            if (i/len(trainloader)*100 % 10) < 0.1:
                print(
                    "[{epoch}, {batch}%] generator loss: {g_loss} discriminator loss: {d_loss}"
                    .format(epoch=epoch+1,batch=int(i/len(trainloader)*100),g_loss=loss_dict["g_loss"].detach(),d_loss=loss_dict["d_loss"].detach())
                )
        print(
            "[{epoch}, {batch}%] generator loss: {g_loss} discriminator loss: {d_loss}"
            .format(epoch=epoch+1,batch=100,g_loss=loss_dict["g_loss"].detach(),d_loss=loss_dict["d_loss"].detach())
        )
        print(
            "[{epoch}, train] generator loss: {g_loss} discriminator loss {d_loss}"
            .format(epoch=epoch+1,g_loss=running_loss_G/(i+1),d_loss=running_loss_D/(i+1))
        )
        net.eval()
        running_loss_G, running_loss_D = 0, 0
        for i, data in enumerate(testloader, start=0):
            inputs, labels = data
            one_hot_labels = torch.nn.functional.one_hot(labels)
            z = torch.randn((batch_size, latent_dim))
            generated_imgs, predicted_labels_g, predicted_labels_d, predicted_labels_r = net(z, data=inputs.to(DEVICE), context=one_hot_labels.to(DEVICE))
            loss_dict = net.loss(predicted_labels_g, predicted_labels_d, predicted_labels_r)

            running_loss_G += loss_dict["g_loss"].detach()
            running_loss_D += loss_dict["d_loss"].detach()

        print(
            "[{epoch}, test] generator loss: {g_loss} discriminator loss: {d_loss}"
            .format(epoch=epoch+1,g_loss=running_loss_G/(i+1),d_loss=running_loss_D/(i+1))
        )
        print()

    labels = torch.tensor([[i for _ in range(n_classes)] for i in range(n_classes)]).view(-1)
    imgs = net.cpu().sample(batch_size=n_classes**2, context=torch.nn.functional.one_hot(labels))
    imshow(torchvision.utils.make_grid(imgs, nrow=n_classes))