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
        "epochs" : 100,
        "batch_size" : 200,
        "n_steps" : 60,
        "step_size" : 10,
        "noise_scale" : 0.005,
        "grad_clip" : 0.03,
        "alpha" : 0.1,
        "grad_norm_clip" : 0.1,
        "learning_rate" : 1e-4,
        "dataset" : "MNIST",
        "architecture" : "CNN",
        "hidden_dims" : [64, 64, 64, 64],
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
    net = EBM(cfg["in_size"], cfg["in_channels"], cfg["noise_scale"], cfg["grad_clip"], cfg["step_size"], cfg["alpha"], cfg["hidden_dims"], device=DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"], betas=(0, 0.999))

    wandb.watch(net, log="all", log_freq=1)

    print(net)
    print()

    #data_buffer = torch.rand(cfg["buffer_size"], cfg["batch_size"], cfg["in_channels"], cfg["in_size"], cfg["in_size"], device=DEVICE) * 2 - 1
    sample_buffer = torch.rand(cfg["buffer_size"], cfg["batch_size"], cfg["in_channels"], cfg["in_size"], cfg["in_size"], device=DEVICE) * 2 - 1
    buffer_sample_distribution =  torch.distributions.binomial.Binomial(cfg["batch_size"], cfg["buffer_sample_rate"])

    # train network
    for epoch in range(cfg["epochs"]):
        net.train()
        running_loss = 0
        running_loss_cd = 0
        running_loss_reg = 0
        running_energy_data = 0
        running_energy_samples = 0
        for i, data_batch in enumerate(trainloader, start=0):
            num_new_samples = buffer_sample_distribution.sample().to(int).item()
            random_samples = torch.rand(num_new_samples, cfg["in_channels"], cfg["in_size"], cfg["in_size"], device=DEVICE) * 2 - 1
            buffer_samples = sample_buffer[i, :cfg["batch_size"] - num_new_samples]
            #buffer_data = data_buffer[i, :cfg["batch_size"] - num_new_samples]

            samples = torch.cat([buffer_samples, random_samples], dim=0).detach().to(DEVICE)
            data, _ = [d.to(DEVICE) for d in data_batch]
            small_noise = torch.randn_like(data) * cfg["noise_scale"]
            data.add_(small_noise).clamp_(min=-1.0, max=1.0)
            #data = torch.cat([data, old_data], dim=0).detach().to(DEVICE)
            #if torch.rand(1) > cfg["buffer_sample_rate"] and epoch > 1:
            #    inputs_pos = data_buffer[i]
            #    inputs_neg = sample_buffer[i]
            #else:
            #    data = [d.to(DEVICE) for d in data]
            #    inputs_pos, labels = data
            #    inputs_neg = torch.randn_like(inputs_pos, requires_grad=True, device=DEVICE)
            #    one_hot_labels = torch.nn.functional.one_hot(labels)
            #for param in net.parameters():
            #    param.grad = None
            
            # langevin dynamics
            samples = net.generate_samples(samples, cfg["n_steps"])

            #data_buffer[i] = inputs_pos
            sample_buffer[i] = samples
        
            optimizer.zero_grad()
            input = torch.cat([data, samples], dim=0)
            energy_data, energy_samples = net(input).chunk(2, dim=0)

            loss, loss_cd, loss_reg = net.loss(energy_data, energy_samples)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg["grad_norm_clip"])
            optimizer.step()

            running_loss += loss.item()
            running_loss_cd += loss_cd.item()
            running_loss_reg += loss_reg.item()
            running_energy_data += energy_data.mean().item()
            running_energy_samples += energy_samples.mean().item()

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
        samples = net.generate_samples(input, cfg["n_steps"])
        sample_grid = torch.zeros(4 * cfg["in_size"], 4 * cfg["in_size"], device=DEVICE)
        for r in range(4):
            for c in range(4):
                sample_grid[r*cfg["in_size"]:(r+1)*cfg["in_size"], c*cfg["in_size"]:(c+1)*cfg["in_size"]] = samples[c+4*r]

        buffer_grid = torch.zeros(4 * cfg["in_size"], 4 * cfg["in_size"], device=DEVICE)
        for r in range(4):
            for c in range(4):
                buffer_grid[r*cfg["in_size"]:(r+1)*cfg["in_size"], c*cfg["in_size"]:(c+1)*cfg["in_size"]] = sample_buffer[0, :16]

        wandb.log({
            "Loss/loss": running_loss/(i+1),
            "Loss/contrastive divergence loss": running_loss_cd/(i+1),
            "Loss/regularization loss": running_loss_reg/(i+1),
            "Energy/data" : running_energy_data/(i+1),
            "Energy/samples" : running_energy_samples/(i+1),
            "Images/samples" : wandb.Image(sample_grid.cpu().numpy()),
            "Images/buffer" : wandb.Image(buffer_grid.cpu().numpy())
            })
    wandb.finish()