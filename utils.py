import torch
import torch.nn as nn
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

def gradient_penalty(disc, real_images, fake_images, step, alpha, device="cpu"):
    bs, channels, height, width = real_images.shape
    eps = torch.rand(bs, 1, 1, 1).to(device).repeat(1, channels, height, width)
    merged_images = real_images * eps + fake_images * (1 - eps)
    merged_predict = disc(merged_images, step=step, alpha=alpha)
    gradient_penalty = torch.autograd.grad(
        inputs=merged_images,
        outputs=merged_predict,
        grad_outputs=torch.ones_like(merged_predict),
        create_graph=True,
        retain_graph=True)[0]
    gradient_penalty = gradient_penalty.view(bs, -1)
    gradient_penalty = gradient_penalty.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_penalty - 1) ** 2)
    return gradient_penalty

def weights_init(layer):
    if type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
        nn.init.kaiming_normal_(layer.weight)
    if type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr * 0.1

def new_dataloader(batch_size, img_size):
    data_path = r"C:\Users\Johnny\Desktop\PROGAN\img_align_celeba"
    data = dset.ImageFolder(root=data_path,
                                transform=
                                transforms.Compose([
                                transforms.Resize((img_size, img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.RandomHorizontalFlip(p=0.15),
                               ]))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader

def large_num_period(num):
    return "{:,}".format(num).replace(",", ".")

def plot_losses(g_losses, d_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses,label="G")
    plt.plot(d_losses,label="D")
    plt.xlabel("iterations (in thousands)")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("losses")
    plt.close()
