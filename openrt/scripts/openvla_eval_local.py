import io
import os
import pickle
import time

import hydra
import imageio
import numpy as np
import requests
from PIL import Image

import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import sys

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


from robot.wrappers.crop_wrapper import CropImageWrapper
from robot.wrappers.resize_wrapper import ResizeImageWrapper
from robot.sim.vec_env.vec_env import make_env
#from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
import wandb
import random
import numpy as np
import omegaconf
import cv2

def hydra_to_dict(cfg):
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    return cfg_dict

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)

def setup_wandb(cfg, name, entity, project):

    run = wandb.init(
        name=name,
        entity=entity,
        project=project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread"),
    )

    return run

#from utils.logger import configure_logger
#from utils.system import get_device, set_gpu_mode

sys.path.append("/home/joel/projects/robomimic_openrt")
sys.path.append("/home/joel/projects/robosuite")

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

def apply_center_crop(image, t_h, t_w):
    """
    Crops the center of the input image to the specified height (t_h) and width (t_w).
    
    Parameters:
    - image (numpy.ndarray): The input image array with shape (H, W, C).
    - t_h (int): The target height of the cropped image.
    - t_w (int): The target width of the cropped image.
    
    Returns:
    - numpy.ndarray: The cropped image with shape (t_h, t_w, C).
    """
    H, W, C = image.shape
    
    # Calculate the center of the image
    center_h, center_w = H // 2, W // 2
    
    # Calculate the cropping box
    crop_top = max(center_h - t_h // 2, 0)
    crop_left = max(center_w - t_w // 2, 0)
    
    # Ensure the crop box does not exceed the image dimensions
    crop_bottom = min(crop_top + t_h, H)
    crop_right = min(crop_left + t_w, W)
    
    # Adjust crop box if the bottom or right exceeds image dimensions
    if crop_bottom - crop_top < t_h:
        crop_top = max(crop_bottom - t_h, 0)
    if crop_right - crop_left < t_w:
        crop_left = max(crop_right - t_w, 0)
    
    # Crop the image
    cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]
    
    return cropped_image


class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], unnorm_key: Optional[str]=None, attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        # Register OpenVLA model to HF AutoClasses (not needed if you pushed model to HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
                                
        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.unnorm_key = unnorm_key

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if 'fullft' in self.openvla_path or 'latent' in self.openvla_path:
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)
        elif os.path.isdir(self.openvla_path) and 'World-Model' not in self.openvla_path:
            with open(Path("/".join(self.openvla_path.split('/')[:-1])) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)
        

    def predict_action(self, image: np.ndarray, instruction: str, image_aug: bool) -> str:

        
        # Run VLA Inference
        prompt = get_openvla_prompt(instruction, self.openvla_path)
        
        if image_aug:
            # for cropping when trained with image augmentation
            import math
            temp_image = np.array(image)  # (H, W, C)
            crop_scale = 0.9
            sqrt_crop_scale = math.sqrt(crop_scale)
            temp_image_cropped = apply_center_crop(
                temp_image, t_h=int(sqrt_crop_scale * temp_image.shape[0]), t_w=int(sqrt_crop_scale * temp_image.shape[1])
            )
            temp_image = Image.fromarray(temp_image_cropped)
        else:
            temp_image = Image.fromarray(image)
        temp_image = temp_image.resize([224, 224], Image.Resampling.BILINEAR)  # IMPORTANT: dlimp uses BILINEAR resize
        image = temp_image
        inputs = self.processor(prompt, image.convert("RGB")).to(self.device, dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)

        return action


        


def call_rt(img, msg, model, cfg):
    # IP address always changes every time the server is rebooted. Please check the text-generation-webui cli interface

    start = time.time()

    # image = Image.fromarray(img)
    # Compress the images using JPEG
    # buffer = io.BytesIO()
    # image.save(buffer, format="JPEG", quality=80)
    # jpeg_data = buffer.getvalue()

    # prompt = f"<image>\nWhat action should the robot take to `{msg}`"
    # data = {"images": jpeg_data, "queries": prompt, "answers": ""}

    action = model.predict_action(img, msg, cfg.image_aug)
    
    print(f"PREDICTION: {action}")

    # action = inverse_discretize(response_txt, minmaxlst)
    # action = np.concatenate((action, [float(response_txt[-1])]))

    print(f"Inference time: {time.time() - start} seconds")

    return action


@hydra.main(
    config_path="../../configs/", config_name="eval_openvla_real", version_base="1.1"
)
def run_experiment(cfg):

    set_random_seed(cfg.seed)
    cfg.robot.max_path_length = cfg.max_episode_length
    assert cfg.robot.imgs, "ERROR: set robot.imgs=true to record image observations!"

    # create env
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    camera_names = [k + "_rgb" for k in env.get_images().keys()]

    # crop image observations
    if cfg.aug.camera_crop is not None:
        env = CropImageWrapper(
            env,
            x_min=cfg.aug.camera_crop[0],
            x_max=cfg.aug.camera_crop[1],
            y_min=cfg.aug.camera_crop[2],
            y_max=cfg.aug.camera_crop[3],
            image_keys=[cn for cn in camera_names],
            crop_render=True,
        )

    # resize image observations
    '''
    if cfg.aug.camera_resize is not None:
        env = ResizeImageWrapper(
            env,
            size=cfg.aug.camera_resize,
            image_keys=[cn for cn in camera_names],
        )
    '''
    data = {
        "obs": [],
        "act": [],
        "rgbd": [],
    }

    env.seed(cfg.seed)
    obs = env.reset()
    
    # Load the model
    model = OpenVLAServer(cfg.openvla_path) 
    print(f'finished loading model')
    count = 0

    done = False
    for i in range(cfg.traj_length):
        print(f'Step: {i}')

        start = time.time()

        img = obs[camera_names[0]]        
        data["rgbd"].append(img)

        # call openrt api
        act = call_rt(img, cfg.msg, model, cfg)

        
        # step
        next_obs, reward, done, info = env.step(act)

        print(
            f"Time {np.around(time.time()-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(act[0],3)}"
        )

        data["act"].append(act)
        data["obs"].append(obs)

        image = next_obs["215122255213_rgb"]
        # cv2.imshow('Real-time video', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if cfg.save_info:
            directory_path = f"{cfg.base_dir}/{cfg.exp_id}/eval/{cfg.folder_name}/{cfg.trial_num}"
            #directory_path = f"{cfg.base_dir}/{cfg.exp_id}/eval/openvla_4500/{cfg.trial_num}"
            if not os.path.exists(directory_path):
                # Create the directory
                os.makedirs(directory_path)
                print(f"Directory '{directory_path}' created.")
            cv2.imwrite(f"{directory_path}/{count}.png",cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        obs = next_obs
        count+=1

    # stack acts and imgs
    for k in ["act", "rgbd"]:
        data[k] = np.stack(data[k])

    data_obs_tmp = {}
    for k in data["obs"][0].keys():
        data_obs_tmp[k] = data["obs"][0][k][None]

    for obs in data["obs"][1:]:
        for k, v in obs.items():
            data_obs_tmp[k] = np.concatenate((data_obs_tmp[k], v[None]))


if __name__ == "__main__":
    run_experiment()
