import os
import glob
from PIL import Image

def create_gif(gif_file_name, img_folder_path):
    """create gif out of all progress images in a folder"""
    # format final path if needed
    if not gif_file_name.endswith(".gif"):
        gif_file_name += ".gif"

    def img_val(img_path):
        """sort fn for image paths"""
        img_path =  img_path.replace("resolution", "")
        resolution = int(img_path.split("x")[0])
        samples = int(img_path.split("-")[1].replace("samples.png", ""))
        val = resolution * 100 + samples // 100_000
        return val

    # load img paths
    try:
        image_paths = os.listdir(img_folder_path)
    except OSError:
        print("The path you entered is invalid")
        return
    end_img_cnt = 10
    
    # sort image paths and bring them into the right order
    image_paths.sort(key=img_val)
    # load images from img paths
    imgs = [Image.open(os.path.join(img_folder_path, f)) for f in image_paths]
    for i in range(end_img_cnt):
        imgs.append(imgs[-1])
    # create and save gif to path
    imgs.pop(0).save(fp=gif_file_name, format='GIF', append_images=imgs,
                    save_all=True, duration=100, loop=0)

if __name__ == "__main__":
    create_gif("progress", r"C:\Users\Johnny\Desktop\PROGAN\intermediate_images\model_1")
