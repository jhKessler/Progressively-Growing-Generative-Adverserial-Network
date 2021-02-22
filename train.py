import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.gen_model import Generator
from models.disc_model import Discriminator
from checkpoint import Checkpoint
from utils import *

# seed for reproducability
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training parameters
batch_size = [64, 64, 64, 32, 16, 4]
# beta coeffs for adam optimizer
betas = (0.0, 0.99)
# random noise dimensions
noise_dim = 256

start_step = 1
max_steps = 4
assert (0 <= start_step <= 5) and (0 <= max_steps <= 5), "Please enter valid step-values (0-5)"

# wgan-gp values
disc_train_count = 3
GRAD_VAL = 10

fade_size = 800_000
phase_size = fade_size * 3

lr = [0.007, 0.01, 0.01, 0.01, 0.01, 0.006]

# set to None if starting a new model, else enter checkpoint id
checkpoint = 2
# set to true when overwriting a checkpoint that you have specified above with a new model
reset = True
assert not (checkpoint is None and reset), "Please specify a valid checkpoint to overwrite"

# model definitions
if checkpoint is None or reset:

	step = start_step
	used_samples = 0

	generator = Generator(noise_dim).to(device)
	g_optimizer = optim.Adam(generator.parameters(), lr=lr[step], betas=betas)

	discriminator = Discriminator().to(device)
	d_optimizer = optim.Adam(discriminator.parameters(), lr=lr[step], betas=betas)
	
	start = time.time()
	dataset = 1 # 1 = CELEBA
	start_iter = 0
	loss_type = 1 # 1 = wgan-gp
	preview_noise = torch.randn(64, noise_dim).to(device)

else:
	checkpoint = Checkpoint(load_id=checkpoint)
	val_dict = checkpoint.give_model_dict()

	step = val_dict["step"]
	used_samples = val_dict["samples"]

	generator = val_dict["generator"]
	g_optimizer = val_dict["g_optimizer"]

	discriminator = val_dict["discriminator"]
	d_optimizer = val_dict["d_optimizer"]
	
	start = time.time() - val_dict["time"]
	dataset = val_dict["dataset"]
	start_iter = val_dict["iteration"]
	loss_type = val_dict["loss_type"]
	preview_noise = val_dict["preview_noise"]

	del val_dict

def train(iterations=5_000_000):
	global step, checkpoint, used_samples, start, loss_type, disc_train_count, dataset

	data = new_dataloader(batch_size[step], get_resolution(step), dataset)
	curr_res = get_resolution(step)

	toggle_grad(discriminator, True)
	toggle_grad(generator, True)

	adjust_lr(d_optimizer, lr[step])
	adjust_lr(g_optimizer, lr[step])
	print(f"Generator-Parameters: {count_parameters(generator)}")
	print(f"Discriminator-Parameters: {count_parameters(discriminator)}")
	print(f"Training on {torch.cuda.get_device_name(device)}")
	print("Starting Training Loop...")
	print(f"Training started with Resolution {curr_res}x{curr_res} and {used_samples} samples used (Time: {(time.time()-start) // 60} minutes)")

	for current_iteration in range(start_iter, iterations):
		gc.collect()

		# switch to next phase after current phase is completed
		if used_samples > phase_size and step != max_steps:
			print(f"Training for Resolution {curr_res}x{curr_res} done after {(time.time()-start) // 60} minutes, Used Samples: {used_samples}")
			used_samples = 0
			step += 1

			# break training loop after last step (64x64) is completed
			if step > max_steps:
				step = max_steps 
				break

			curr_res = get_resolution(step)
			adjust_lr(d_optimizer, lr[step])
			adjust_lr(g_optimizer, lr[step])

			torch.cuda.empty_cache()
			data = new_dataloader(batch_size[step], curr_res, dataset)
			loader = iter(data)

		if loss_type == 2:
			toggle_grad(discriminator, True)
			toggle_grad(generator, False)
		
		alpha = min([1, (used_samples + 1) /  fade_size]) if step > start_step else 1

		try:
			batch = next(loader)
		except (NameError, StopIteration):
			loader = iter(data)
			batch = next(loader)

		# train discriminator on real images
		discriminator.zero_grad()
		real_images = batch[0].to(device).float()
		current_bs = real_images.shape[0]
		real_images.requires_grad = True
		real_predict = discriminator(real_images, step=step, alpha=alpha)

		# random noise vector sampling values of gaussian distribution
		gen_noise = torch.randn(current_bs, noise_dim).to(device)
		gen_imgs = generator(gen_noise, step=step, alpha=alpha)
		fake_predict = discriminator(gen_imgs, step=step, alpha=alpha)
		
		# r1 loss
		if loss_type == 2:
			real_loss = F.softplus(-real_predict.mean())
			real_loss.backward(retain_graph=True)
			gp = r1_gradient_penalty(discriminator, real_predict, real_images, step, alpha, device=device)
			gp.backward()

			# train discriminator on fake images
			fake_loss = F.softplus(fake_predict.mean())
			fake_loss.backward()

		# wgan-gp loss
		elif loss_type == 1:
			gp = wgan_gradient_penalty(discriminator, real_images, gen_imgs, step, device=device, alpha=alpha)
			disc_loss = -(real_predict.mean() - fake_predict.mean()) + (GRAD_VAL * gp)

			discriminator.zero_grad()
			if current_iteration % disc_train_count == 0:
				disc_loss.backward(retain_graph=True)
			else:
				disc_loss.backward()

		d_optimizer.step()
		used_samples += current_bs

		# r1 loss
		if loss_type == 2:
			toggle_grad(generator, True)
			toggle_grad(discriminator, False)

			# do another forward pass on fake images to train generator
			gen_noise = torch.randn(current_bs, noise_dim).to(device)
			gen_imgs = generator(gen_noise, step=step, alpha=alpha)
			gen_predict = discriminator(gen_imgs, step=step, alpha=alpha).mean()
			gen_loss = F.softplus(-gen_predict)
			generator.zero_grad()
			gen_loss.backward()
			g_optimizer.step()

		# wgan-gp loss
		elif loss_type == 1 and current_iteration % disc_train_count == 0:
			gen_predict = discriminator(gen_imgs, step=step, alpha=alpha).mean()

			# g loss - maximize d(f)
			gen_loss = -gen_predict
			generator.zero_grad()
			gen_loss.backward()
			g_optimizer.step()

		if used_samples % 100_000 < current_bs:
			if used_samples % (phase_size // 8) < current_bs:
				generate_and_save_images(used_samples, preview_noise, generator, alpha, step)
			samples = format_large_nums(used_samples)
			iter_nr = format_large_nums(current_iteration)
			print(f"[{iter_nr}] Resolution: {curr_res}x{curr_res}, Samples: {samples}, alpha: {alpha}, Time: {(time.time()-start) // 60} minutes")

			# save progress
			model_dict = {
					"generator" : generator,
					"g_optimizer" : g_optimizer,
					"discriminator" : discriminator,
					"d_optimizer" : d_optimizer,
					"step" : step, 
					"iteration" : current_iteration,
					"samples" : used_samples,
					"time" : (time.time()-start),
					"preview_noise" : preview_noise,
					"loss_type" : loss_type,
					"dataset" : dataset
							}

			if checkpoint is None:
				checkpoint = Checkpoint(save_dict=model_dict)
			else:
				if type(checkpoint) == int:
					checkpoint = Checkpoint(load_id=checkpoint)
				checkpoint.save(model_dict)
			del model_dict


if __name__ == "__main__":
	train()