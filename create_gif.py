import os
import glob
from PIL import Image

def create_gif(gif_file_name, img_folder_path=r"C:\Users\Johnny\Desktop\PROGAN\intermediate_images"):
    # format final path if needed
    if not gif_file_name.endswith(".gif"):
        gif_file_name += ".gif"

    # load img paths
    try:
        image_paths = [str(image_path) for image_path in os.listdir(img_folder_path)]
    except OSError:
        print("The path you entered is invalid")
        return
    # sort image paths and bring them into the right order
    image_paths.sort(key=lambda x: (int("".join([num for num in x if num.isdigit()]))))
    # load images from img paths
    imgs = [Image.open(os.path.join(img_folder_path, f)) for f in image_paths]
    start_img = imgs.pop(0)
    # create and save gif to path
    start_img.save(fp=gif_file_name, format='GIF', append_images=imgs,
                    save_all=True, duration=100, loop=0)
