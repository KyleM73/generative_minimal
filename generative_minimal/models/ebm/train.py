import torch
import torchvision
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # mps is almost always slower
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True # enables cuDNN auto-tuner
torch.manual_seed(0)

from generative_minimal import ROOT_DIR
from generative_minimal.models import EBM
from generative_minimal import utils

if __name__ == "__main__":

    cfg = {
        "in_channels" : 1,
        "in_size" : 28,
        "epochs" : 10000,
        "batch_size" : 200,
        "n_steps" : 20,
        "step_size" : 0.5,
        "noise_scale" : 1.0,
        "alpha" : 1,
        "learning_rate" : 2e-4,
        "dataset" : "MNIST",
        "architecture" : "CNN",
        "hidden_dims" : [64, 64, 64],
        "buffer_size" : 300,
        "buffer_sample_rate" : 0.05,
    }

    wandb.init(
        # set the wandb project where this run will be logged
        project="ebm-mnist",

        # track hyperparameters and run metadata
        config=cfg
    )
    # MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # from PIL.Image.Image to torch.Tensor
        torchvision.transforms.Normalize((0.5), (0.5)), # from [0,1] to [-1,1]
        ]) 
    trainset = torchvision.datasets.MNIST(root="{}/data".format(ROOT_DIR), train=True, download=False, transform=transform) # 60k
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    n_classes = len(classes)

    # make dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg["batch_size"],
                                            shuffle=True, num_workers=4, pin_memory=True)
    
    # define network
    net = EBM(cfg["in_size"], cfg["in_channels"], cfg["noise_scale"], cfg["step_size"], cfg["alpha"], cfg["hidden_dims"], device=DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])

    wandb.watch(net, log_freq=1)

    print(net)
    print()

    pos_buffer = torch.zeros(cfg["buffer_size"], cfg["batch_size"], cfg["in_channels"], cfg["in_size"], cfg["in_size"], device=DEVICE)
    neg_buffer = torch.zeros(cfg["buffer_size"], cfg["batch_size"], cfg["in_channels"], cfg["in_size"], cfg["in_size"], device=DEVICE)

    # train network
    for epoch in range(cfg["epochs"]):
        net.train()
        running_loss = 0
        for i, data in enumerate(trainloader, start=0):
            if torch.rand(1) > cfg["buffer_sample_rate"] and epoch > 1:
                inputs_pos = pos_buffer[i]
                inputs_neg = neg_buffer[i]
            else:
                data = [d.to(DEVICE) for d in data]
                inputs_pos, labels = data
                inputs_neg = torch.randn_like(inputs_pos, requires_grad=True, device=DEVICE)
                #one_hot_labels = torch.nn.functional.one_hot(labels)
            #for param in net.parameters():
            #    param.grad = None
            
            # langevin dynamics
            inputs_neg = net.sample(inputs_neg.requires_grad_(True), cfg["n_steps"])

            pos_buffer[i] = inputs_pos
            neg_buffer[i] = inputs_neg
        
            optimizer.zero_grad()
            energy_pos = net(inputs_pos)
            energy_neg = net(inputs_neg)

            loss = net.loss(energy_pos, energy_neg)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
            optimizer.step()

            running_loss += loss.item()

            if (i/len(trainloader)*100 % 10) < 0.1:
                print(
                    "[{epoch}, {batch}%] loss: {loss}".format(epoch=epoch+1,batch=int(i/len(trainloader)*100),loss=loss.item())
                )
        print(
            "[{epoch}, {batch}%] loss: {loss}".format(epoch=epoch+1,batch=100,loss=loss.item())
        )
        print(
            "[{epoch}] loss: {loss}".format(epoch=epoch+1,loss=running_loss/(i+1))
        )
        # get samples for eval
        input = torch.randn(16, 1, cfg["in_size"], cfg["in_size"], requires_grad=True, device=DEVICE)
        samples = net.sample(input, cfg["n_steps"])
        sample_grid = torch.zeros(4 * cfg["in_size"], 4 * cfg["in_size"], device=DEVICE)
        for r in range(4):
            for c in range(4):
                sample_grid[r*cfg["in_size"]:(r+1)*cfg["in_size"], c*cfg["in_size"]:(c+1)*cfg["in_size"]] = samples[c+4*r]

        wandb.log({"loss": running_loss/(i+1), "samples" : wandb.Image(sample_grid.cpu().numpy())})
    wandb.finish()