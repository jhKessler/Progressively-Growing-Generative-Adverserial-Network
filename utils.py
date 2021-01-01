import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torchvision.utils as vutils
from torchvision import transforms
import matplotlib.pyplot as plt

# gradient penalty for discriminator based on wgan-gp paper "Improved Training of Wasserstein GANs" (https://arxiv.org/abs/1704.00028)
def gradient_penalty(disc, real_images, fake_images, step, alpha, device="cpu"):
    bs, channels, height, width = real_images.shape
    eps = torch.rand(bs, 1, 1, 1).to(device).repeat(1, channels, height, width)
    # merge fake and real images
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
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=3)
    return loader

def format_large_nums(num):
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

# generate images when training is completed
def generate_final_images(model, alpha, noise_dim, num=128):
    image_folder = r"C:\Users\Johnny\Desktop\PROGAN\final_images"

    with torch.no_grad():
        noise = torch.randn(num, noise_dim).cuda()
        images = model(noise, step=step, alpha=alpha).detach().cpu()

    for i in range(images.shape[0]):
        image = images[i]
        save_image(image,
                   os.path.join(image_folder, f"image{i}.jpg"),
                   normalize=True,
                   range=(-1, 1))

# generate intermediate imgs for progress gif
def generate_and_save_images(iteration, noise, model, alpha, step):
    fake_folder = r"C:\Users\Johnny\Desktop\PROGAN\intermediate_images"
    fake_img_path = os.path.join(fake_folder, f"iteration{iteration}resolution{get_resolution(step)}x{get_resolution(step)}")

    with torch.no_grad():
        images = model(noise, step=step, alpha=alpha).detach().cpu()
        images = np.transpose(vutils.make_grid(images, padding=2, normalize=True), (1,2,0))

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(images)
    plt.savefig(fake_img_path)
    plt.close()

def get_resolution(step):
    return 4 * (2**step)