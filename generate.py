import torch
from checkpoint import Checkpoint
from models.generator import Generator
from utils import generate_final_images

def generate(num_images_to_generate, checkpoint_to_load):
    checkpoint = Checkpoint(load_id=checkpoint_to_load)
    generator = Checkpoint.give_model_dict()["generator"]

    generate_final_images(generator, generator.noise_dim, num=num_images_to_generate)