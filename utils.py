import os
import subprocess
import numpy as np
import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torchvision.utils as vutils
from torchvision import transforms
from models.generator import Generator
from models.discriminator import Discriminator
import matplotlib.pyplot as plt

# gradient pentalty for gan loss fn
def gan_gradient_penalty(disc, real_predict, real_images, step, alpha, device="cpu"):
    grad_real = torch.autograd.grad(outputs=real_predict.sum(),
                         inputs=real_images,
                         create_graph=True)[0]
    grad_penalty = (grad_real.view(grad_real.size(0),
                                    -1).norm(2, dim=1)**2).mean()
    grad_penalty = 10 / 2 * grad_penalty
    return grad_penalty

# gradient penalty for discriminator based on wgan-gp paper "Improved Training of Wasserstein GANs" (https://arxiv.org/abs/1704.00028)
def wgan_gradient_penalty(disc, real_images, fake_images, step, alpha, device="cpu"):
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

# count trainable parameters of a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# adjust learning rate of optimizer
def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr * 0.1

# create new dataloader object
def new_dataloader(batch_size, img_size, dataset):
    data_path = r"C:\Users\Johnny\Desktop\PROGAN\img_align_celeba" if dataset == 1 else r"C:\Users\Johnny\Desktop\PROGAN\generative-dog-images\cropped"
    data = dset.ImageFolder(root=data_path,
                                transform=
                                transforms.Compose([
                                transforms.Resize((img_size, img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.RandomHorizontalFlip(p=0.15 if dataset == 1 else 0.3),
                               ]))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=3)
    return loader

# add dots in large numbers to make them more readable
def format_large_nums(num):
    return "{:,}".format(num).replace(",", ".")

# generate intermediate imgs for progress gif
def generate_and_save_images(samples, noise, model, alpha, step, cp_id):
    fake_folder = os.path.join(r"C:\Users\Johnny\Desktop\PROGAN\intermediate_images", f"model_{cp_id}")
    fake_img_path = os.path.join(fake_folder, f"resolution{get_resolution(step)}x{get_resolution(step)}-{samples}samples")

    with torch.no_grad():
        images = model(noise, step=step, alpha=alpha).detach().cpu()
        images = np.transpose(vutils.make_grid(images, padding=2, normalize=True), (1,2,0))

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Progress")
    plt.imshow(images)
    try:
        plt.savefig(fake_img_path)
    except OSError:
        os.mkdir(fake_folder)
        plt.savefig(fake_img_path)
    plt.close()

    # update bot
    bot_args = "-Dexec.args=\"" + fake_img_path.replace("\\", "/") + ".png" + "\""
    subprocess.Popen(["mvn", "exec:java", "-Dexec.mainClass=AiStatusBot", bot_args, "-f", r"C:\Users\Johnny\IdeaProjects\AiStatusBot\pom.xml"], 
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)

# generate images when training is completed
def generate_final_images(model, noise_dim, num=128):
    image_folder = r"C:\Users\Johnny\Desktop\PROGAN\final_images"

    with torch.no_grad():
        noise = torch.randn(num, noise_dim).cuda()
        images = model(noise, step=4, alpha=1).detach().cpu()

    for i in range(images.shape[0]):
        image = images[i]
        save_image(image,
                   os.path.join(image_folder, f"image{i}.jpg"),
                   normalize=True,
                   range=(-1, 1))

# return resolution of current step
def get_resolution(step):
    return 4 * (2**step)

# turn gradient of model on/off
def toggle_grad(model, mode):
    for p in model.parameters():
        p.requires_grad = mode