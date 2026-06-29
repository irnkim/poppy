import os
import numpy as np
import torch
from utils.helpers import *


def convert_to_data(imgs, device):
    for i in range(len(imgs)):
        img = imgs[i]
        if img is None:
            imgs[i] = img
            continue
        if img.shape[-1] == 3:
            img = img.transpose((2, 0, 1))
        else:
            img = np.expand_dims(img, axis=0)
        imgs[i] = torch.from_numpy(img).unsqueeze(0).to(device)
    return imgs


def get_files(in_dir):
    file_names = os.listdir(os.path.join(in_dir, "pol000"))
    file_names = [f.removesuffix(".png") for f in file_names]
    return file_names


def load_files(in_dir, name, device):
    dir_map = {
        "pol000": "pol000",
        "pol045": "pol045",
        "pol090": "pol090",
        "pol135": "pol135",
        "mask":   "mask",
        "normal": "normal",
    }
    in_imgs = {}
    for dirname, var_name in dir_map.items():
        file_path = os.path.join(in_dir, dirname, name)
        try:
            if dirname == "normal":
                image = read_normal(file_path + ".png")
            elif dirname == "mask":
                image = read_mask(file_path + ".png")
            else:
                image = read_img(file_path + ".png")
        except Exception as e:
            if dirname == "normal":
                print("Normal does not exist. Metrics will not be computed.")
            else:
                print(f"Error opening or converting {file_path}: {e}")
            image = None
        in_imgs[var_name] = image
    I_obs, Q_obs, U_obs, phi_obs, rho_obs = calculate_observed_polarization(
        I_0=in_imgs["pol000"],
        I_pi_4=in_imgs["pol045"],
        I_pi_2=in_imgs["pol090"],
        I_3pi_4=in_imgs["pol135"],
    )
    corrected = np.clip(I_obs, 0.0, 1.0)
    imgs = [
        corrected, I_obs, Q_obs, U_obs, phi_obs, rho_obs,
        in_imgs["mask"], in_imgs["normal"],
    ]
    imgs = convert_to_data(imgs, device)
    return imgs


