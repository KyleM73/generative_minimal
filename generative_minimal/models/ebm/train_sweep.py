import torch
import torchvision
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # mps is almost always slower
if DEVICE == "cuda": torch.backends.cudnn.benchmark = True # enables cuDNN auto-tuner
torch.manual_seed(0)

from generative_minimal import ROOT_DIR
from generative_minimal.models import EBM
from generative_minimal import utils

sweep_config = {
    "method" : "bayes", #options: "random", "bayes", "grid" (bayes requires an additional config, "metric" to be set; see below)
    "name" : "sweep",
    "metric" : {"goal" : "minimize", "name" : "Loss/loss"}, #options: "minimize", "maximize", "target" (target requires an additional param, "target", to be set)
    "run_cap" : 500, #max number of sweeps to run
    "parameters" : {
        "batch_size": {"distribution": "q_uniform", "min": 50, "max": 1000, "q": 10},
        "epochs": {"distribution": "q_uniform", "min": 100, "max": 1000, "q": 100},
        "learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0.00001},
        "n_steps": {"distribution": "q_uniform", "min": 10, "max": 100, "q": 10},
        "step_size": {"distribution": "q_uniform", "max": 100, "min": 1, "q": 1},
        "noise_scale": {"distribution": "uniform", "max": 0.1, "min": 0.0001},
        "grad_clip": {"distribution": "uniform", "max": 1.0, "min": 0.001},
        "grad_norm_clip": {"distribution": "uniform", "max": 1.0, "min": 0.001},
        "buffer_sample_rate": {"distribution": "q_uniform", "max": 1.0, "min": 0.0, "q": 0.01},
        "alpha": {"distribution": "uniform", "max": 1.0, "min": 0.01},
    }
}

def main():
    cfg = {
        "in_channels" : 1,
        "in_size" : 28,
        #"epochs" : 200,
        #"batch_size" : 200,
        #"n_steps" : 60,
        "n_test_steps" : 256,
        #"step_size" : 10,
        #"noise_scale" : 0.005,
        #"grad_clip" : 0.03,
        #"alpha" : 0.1,
        #"grad_norm_clip" : 0.1,
        #"learning_rate" : 1e-4,
        "dataset" : "MNIST",
        "architecture" : "CNN",
        "hidden_dims" : [16, 32, 64],
        "buffer_size" : 300,
        #"buffer_sample_rate" : 0.05,
    }
    wandb.init(
        # set the wandb project where this run will be logged
        project="ebm-mnist-sweep",

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=wandb.config.batch_size,
                                            shuffle=True, num_workers=4, pin_memory=True)
    
    # define network
    net = EBM(cfg["in_size"], cfg["in_channels"], wandb.config.noise_scale, wandb.config.grad_clip, wandb.config.step_size, wandb.config.alpha, cfg["hidden_dims"], device=DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=wandb.config.learning_rate)

    wandb.watch(net, log="all", log_freq=1)

    print(net)
    print()

    #data_buffer = torch.rand(cfg["buffer_size"], wandb.config.batch_size, cfg["in_channels"], cfg["in_size"], cfg["in_size"], device=DEVICE) * 2 - 1
    sample_buffer = torch.rand(cfg["buffer_size"], wandb.config.batch_size, cfg["in_channels"], cfg["in_size"], cfg["in_size"], device=DEVICE) * 2 - 1
    buffer_sample_distribution =  torch.distributions.binomial.Binomial(wandb.config.batch_size, wandb.config.buffer_sample_rate)

    # train network
    for epoch in range(wandb.config.epochs):
        net.train()
        running_loss = 0
        running_loss_cd = 0
        running_loss_reg = 0
        running_energy_data = 0
        running_energy_samples = 0
        for i, data_batch in enumerate(trainloader, start=0):
            num_new_samples = buffer_sample_distribution.sample().to(int).item()
            random_samples = torch.rand(num_new_samples, cfg["in_channels"], cfg["in_size"], cfg["in_size"], device=DEVICE) * 2 - 1
            buffer_samples = sample_buffer[i, :wandb.config.batch_size - num_new_samples]
            #buffer_data = data_buffer[i, :wandb.config.batch_size - num_new_samples]

            samples = torch.cat([buffer_samples, random_samples], dim=0).detach().to(DEVICE)
            data, _ = [d.to(DEVICE) for d in data_batch]
            #data = torch.cat([buffer_data, data[:num_new_samples]], dim=0).detach().to(DEVICE)
            small_noise = torch.randn_like(data) * wandb.config.noise_scale
            data.add_(small_noise).clamp_(min=-1.0, max=1.0)
            
            # langevin dynamics
            samples = net.generate_samples(samples, wandb.config.n_steps)

            #data_buffer[i] = data
            sample_buffer[i] = samples
        
            optimizer.zero_grad()
            input = torch.cat([data, samples], dim=0)
            energy_data, energy_samples = net(input).chunk(2, dim=0)

            loss, loss_cd, loss_reg = net.loss(energy_data, energy_samples)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=wandb.config.grad_norm_clip)
            optimizer.step()

            running_loss += loss.item()
            running_loss_cd += loss_cd.item()
            running_loss_reg += loss_reg.item()
            running_energy_data += -energy_data.mean().item()
            running_energy_samples += -energy_samples.mean().item()

        print(
            "[{epoch}] loss: {loss}".format(epoch=epoch+1,loss=running_loss/(i+1))
        )
        # get samples for eval
        input = torch.randn(16, 1, cfg["in_size"], cfg["in_size"], requires_grad=True, device=DEVICE)
        samples = net.generate_samples(input, cfg["n_test_steps"])
        sample_grid = torch.zeros(4 * cfg["in_size"], 4 * cfg["in_size"], device=DEVICE)
        for r in range(4):
            for c in range(4):
                sample_grid[r*cfg["in_size"]:(r+1)*cfg["in_size"], c*cfg["in_size"]:(c+1)*cfg["in_size"]] = samples[c+4*r]

        buffer_grid = torch.zeros(4 * cfg["in_size"], 4 * cfg["in_size"], device=DEVICE)
        for r in range(4):
            for c in range(4):
                buffer_grid[r*cfg["in_size"]:(r+1)*cfg["in_size"], c*cfg["in_size"]:(c+1)*cfg["in_size"]] = sample_buffer[0, c+4*r]

        #data_grid = torch.zeros(4 * cfg["in_size"], 4 * cfg["in_size"], device=DEVICE)
        #for r in range(4):
        #    for c in range(4):
        #        data_grid[r*cfg["in_size"]:(r+1)*cfg["in_size"], c*cfg["in_size"]:(c+1)*cfg["in_size"]] = data[c+4*r]

        wandb.log({
            "Loss/loss": running_loss/(i+1),
            "Loss/contrastive divergence loss": running_loss_cd/(i+1),
            "Loss/regularization loss": running_loss_reg/(i+1),
            "Energy/data" : running_energy_data/(i+1),
            "Energy/samples" : running_energy_samples/(i+1),
            "Images/samples" : wandb.Image(sample_grid.cpu().numpy()),
            "Images/buffer" : wandb.Image(buffer_grid.cpu().numpy()),
            #"Images/data" : wandb.Image(data_grid.cpu().numpy()),
            })

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project="ebm-mnist-sweep")
    wandb.agent(sweep_id, function=main, count=sweep_config["run_cap"])