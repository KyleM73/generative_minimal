import torch
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # mps is almost always slower
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True # enables cuDNN auto-tuner
torch.manual_seed(0)

from generative_minimal import ROOT_DIR
from generative_minimal.models import GAN, DCGAN
from generative_minimal import utils

if __name__ == "__main__":
    # MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # from PIL.Image.Image to torch.Tensor
        torchvision.transforms.Normalize((0.5), (0.5)), # from [0,1] to [-1,1]
        ]) 
    trainset = torchvision.datasets.MNIST(root="{}/data".format(ROOT_DIR), train=True, download=False, transform=transform) # 60k
    testset = torchvision.datasets.MNIST(root="{}/data".format(ROOT_DIR), train=False, download=False, transform=transform) # 10k
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    n_classes = len(classes)
    in_channels = 1
    in_size = 28
    latent_dim = 128
    epochs = 100
    batch_size = 64
    clip_value = 0.01
    n_critic = 5

    # make dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, pin_memory=True)
    
    # define network
    net = GAN(in_size=in_size, in_channels=in_channels, latent_dim=latent_dim, context_dim=n_classes, device=DEVICE)
    optimizer_G = torch.optim.RMSprop(net.generator.parameters(), lr=5e-5)
    optimizer_D = torch.optim.RMSprop(net.discriminator.parameters(), lr=5e-5)

    print(net)
    print()

    # train network
    for epoch in range(epochs):
        net.train()
        running_loss_G, running_loss_D = 0, 0
        for i, data in enumerate(trainloader, start=0):
            data = [d.to(DEVICE) for d in data]
            inputs, labels = data

            for param in net.discriminator.parameters():
                param.grad = None

            generated_images = net.sample(batch_size=inputs.size(0)).detach()
            discrimminator_loss = -torch.mean(net.discriminate(inputs)) + torch.mean(net.discriminate(generated_images))
            discrimminator_loss.backward()

            optimizer_D.step()

            for param in net.discriminator.parameters():
                param.data.clamp_(-clip_value, clip_value)

            if i % n_critic == 0:
                for param in net.generator.parameters():
                    param.grad = None

                generated_images = net.sample(batch_size=inputs.size(0))
                generator_loss = -torch.mean(net.discriminate(generated_images))
                generator_loss.backward()

                optimizer_G.step()

                running_loss_G += generator_loss.detach()
            running_loss_D += discrimminator_loss.detach()

            if (i/len(trainloader)*100 % 10) < 0.1:
                print(
                    "[{epoch}, {batch}%] generator loss: {g_loss} discriminator loss: {d_loss}"
                    .format(epoch=epoch+1,batch=int(i/len(trainloader)*100),g_loss=generator_loss.detach(),d_loss=discrimminator_loss.detach())
                )
        print(
            "[{epoch}, {batch}%] generator loss: {g_loss} discriminator loss: {d_loss}"
            .format(epoch=epoch+1,batch=100,g_loss=generator_loss.detach(),d_loss=discrimminator_loss.detach())
        )
        print(
            "[{epoch}, train] generator loss: {g_loss} discriminator loss {d_loss}"
            .format(epoch=epoch+1,g_loss=n_critic*running_loss_G/(i+1),d_loss=running_loss_D/(i+1))
        )
        net.eval()
        running_loss_G, running_loss_D = 0, 0
        for i, data in enumerate(testloader, start=0):
            data = [d.to(DEVICE) for d in data]
            inputs, labels = data

            generated_images = net.sample(batch_size=inputs.size(0)).detach()
            discrimminator_loss = -torch.mean(net.discriminate(inputs)) + torch.mean(net.discriminate(generated_images))

            generated_images = net.sample(batch_size=inputs.size(0))
            generator_loss = -torch.mean(net.discriminate(generated_images))

            running_loss_G += generator_loss.detach()
            running_loss_D += discrimminator_loss.detach()
        print(
            "[{epoch}, test] generator loss: {g_loss} discriminator loss: {d_loss}"
            .format(epoch=epoch+1,g_loss=running_loss_G/(i+1),d_loss=running_loss_D/(i+1))
        )
        print()

    labels = torch.tensor([[i for _ in range(n_classes)] for i in range(n_classes)], device=DEVICE).view(-1)
    imgs = net.sample(batch_size=n_classes**2)
    utils.imshow(torchvision.utils.make_grid(imgs.to("cpu"), nrow=n_classes))