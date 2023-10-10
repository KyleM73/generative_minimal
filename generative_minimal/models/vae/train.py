import torch
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # mps is almost always slower
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True # enables cuDNN auto-tuner
torch.manual_seed(0)

from generative_minimal import ROOT_DIR
from generative_minimal.models import VAE, CVAE
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
    net = VAE(in_size=in_size, in_channels=in_channels, latent_dim=latent_dim, context_dim=n_classes, device=DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    print(net)
    print()

    # train network
    for epoch in range(epochs):
        net.train()
        running_loss, running_recons, running_kld = 0, 0, 0
        for i, data in enumerate(trainloader, start=0):
            data = [d.to(DEVICE) for d in data]
            inputs, labels = data
            one_hot_labels = torch.nn.functional.one_hot(labels)
            for param in net.parameters():
                param.grad = None
            generated, src, mu, logvar = net(inputs, context=one_hot_labels)
            loss_dict = net.loss(src, generated, mu, logvar, kld_weight_train)
            loss_dict["loss"].backward()
            optimizer.step()

            running_loss += loss_dict["loss"].detach()
            running_recons += loss_dict["reconstruction"]
            running_kld += loss_dict["kld"]

            if (i/len(trainloader)*100 % 10) < 0.1:
                print(
                    "[{epoch}, {batch}%] loss: {loss} reconstruction loss: {recons} kld loss: {kld}"
                    .format(epoch=epoch+1,batch=int(i/len(trainloader)*100),loss=loss_dict["loss"].detach(),recons=loss_dict["reconstruction"],kld=loss_dict["kld"])
                )
        print(
            "[{epoch}, {batch}%] loss: {loss} reconstruction loss: {recons} kld loss: {kld}"
            .format(epoch=epoch+1,batch=100,loss=loss_dict["loss"].detach(),recons=loss_dict["reconstruction"],kld=loss_dict["kld"])
        )
        print(
            "[{epoch}, train] loss: {loss} reconstruction loss: {recons} kld loss: {kld}"
            .format(epoch=epoch+1,loss=running_loss/(i+1),recons=running_recons/(i+1),kld=running_kld/(i+1))
        )
        net.eval()
        running_loss, running_recons, running_kld = 0, 0, 0
        for i, data in enumerate(testloader, start=0):
            data = [d.to(DEVICE) for d in data]
            inputs, labels = data
            one_hot_labels = torch.nn.functional.one_hot(labels)
            generated, src, mu, logvar = net(inputs, context=one_hot_labels)
            loss_dict = net.loss(src, generated, mu, logvar, kld_weight_test)

            running_loss += loss_dict["loss"].detach()
            running_recons += loss_dict["reconstruction"]
            running_kld += loss_dict["kld"]
        print(
            "[{epoch}, test] loss: {loss} reconstruction loss: {recons} kld loss: {kld}"
            .format(epoch=epoch+1,loss=running_loss/(i+1),recons=running_recons/(i+1),kld=running_kld/(i+1))
        )
        print()

    labels = torch.tensor([[i for _ in range(n_classes)] for i in range(n_classes)], device=DEVICE).view(-1)
    imgs = net.sample(batch_size=n_classes**2, context=torch.nn.functional.one_hot(labels))
    utils.imshow(torchvision.utils.make_grid(imgs.to("cpu"), nrow=n_classes))