import os
import time
import gc

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.utils import save_image, make_grid
import torchvision.utils as vutils

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models.gen_model import Generator
from models.disc_model import Discriminator
from utils import *

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training parameters
batch_size = [256, 256, 128, 64, 64]
betas = (0.0, 0.99)
noise_dim = 256
step = 1
max_steps = 4
fade_size = 800_000
lr = [0.001, 0.002, 0.004, 0.006, 0.008]
phase_size = 2_500_000
GRAD_VAL = 10
# training count of discriminator in respect to generator
disc_train_count = 4


# model definitions
generator = Generator(noise_dim).to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=lr[step], betas=betas)

discriminator = Discriminator().to(device)
#discriminator.apply(weights_init)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr[step], betas=betas)
        
# fixed noise for showing intermediate training progress
val_noise = torch.randn(256, noise_dim).cuda()
preview_noise = torch.randn(64, noise_dim).cuda()

# calculate relative strength of discriminator and generator
@torch.no_grad()
def calcMFID(real_images):
    real_predict = discriminator(real_images, step=step, alpha=alpha).mean()
    val_images = generator(val_noise, step=step, alpha=alpha)
    val_predict = discriminator(val_images, step=step, alpha=alpha).mean()
    MFID_score = real_predict - val_predict
    return MFID_score.item()

def train(iterations=1_000_000):
    global step, alpha

    g_losses = []
    d_losses = []
    data = new_dataloader(batch_size[step], get_resolution(step))
    used_samples = 0
    start = time.time()
    # fixed img batch for comparability
    mfid_batch = next(iter(data))[0].cuda().float()
    step_start = time.time()
    
    print(f"Training on {torch.cuda.get_device_name(device)}")
    print("Starting Training Loop...")
    for current_iteration in range(iterations):
            curr_res = get_resolution(step)
            # current fade value for last block
            alpha = min([1, (used_samples + 1) /  fade_size]) if step > 1 else 1
            # switch to next phase after current phase is completed
            if used_samples > phase_size:
                step_time_taken = (time.time() - step_start) // 60
                step_start = time.time()
                print(f"Time taken for resolution {curr_res}x{curr_res} is {step_time_taken} minutes, Used Samples: {used_samples}, loss_MFID: {calcMFID(mfid_batch)}")
                generate_and_save_images(current_iteration, preview_noise, generator, alpha, step)
                used_samples = 0
                step += 1
                # break training loop after last step (64x64) is completed
                if step > max_steps:
                    step = max_steps 
                    break
                adjust_lr(d_optimizer, lr[step])
                adjust_lr(g_optimizer, lr[step])
                data = new_dataloader(batch_size[step], curr_res)
                mfid_batch = next(iter(data))[0].cuda().float()
                loader = iter(data)

            try:
                batch = next(loader)
            except (NameError, StopIteration):
                loader = iter(data)
                batch = next(loader)

            # train discriminator on real images
            real_images = batch[0].cuda().float()
            current_bs = real_images.shape[0]
            real_predict = discriminator(real_images, step=step, alpha=alpha).mean()

            # random noise vector sampling values of gaussian distribution
            gen_noise = torch.randn(current_bs, noise_dim).cuda()
            gen_imgs = generator(gen_noise, step=step, alpha=alpha)

            fake_predict = discriminator(gen_imgs, step=step, alpha=alpha).mean()

            # wgan-gp loss for discriminator - maximize (d(r) - d(f)) -> wasserstein distance
            gp = gradient_penalty(discriminator, real_images, gen_imgs, step, device=device, alpha=alpha)
            disc_loss = -(real_predict - fake_predict) + (GRAD_VAL * gp)

            discriminator.zero_grad()
            if current_iteration % disc_train_count == 0:
                disc_loss.backward(retain_graph=True)
            else:
                disc_loss.backward()
            d_optimizer.step()
            
            if current_iteration % disc_train_count == 0:
                used_samples += current_bs
                # do another forward pass on fake images to train generator
                gen_predict = discriminator(gen_imgs, step=step, alpha=alpha).mean()
                # g loss - maximize d(f)
                gen_loss = -gen_predict
                generator.zero_grad()
                gen_loss.backward()
                g_optimizer.step()

            if current_iteration % 1_000 == 0:
                g_losses.append((real_predict - fake_predict).detach().item())
                d_losses.append(disc_loss.detach().item())
                if current_iteration % 5_000 == 0:
                    generate_and_save_images(current_iteration, preview_noise, generator, alpha, step)
                    plot_losses(g_losses, d_losses)
                    mfid = calcMFID(mfid_batch) 
                    samples = format_large_nums(used_samples)
                    iter_nr = format_large_nums(current_iteration)
                    print(f"[{iter_nr}] Resolution: {curr_res}x{curr_res}, loss_MFID: {mfid}, Samples: {samples}, alpha: {alpha}, Time: {(time.time()-start) // 60} minutes")
    print(f"Training took {(time.time()-start) // 60} minutes.")

if __name__ == "__main__":
    train()
    generate_final_images(generator, alpha, noise_dim, num=1000)