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

import draccus
# import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

import sys
sys.path.append("/home/joel/projects/robomimic_openrt")
sys.path.append("/home/joel/projects/robosuite")
sys.path.append("/home/joel/projects/polymetis_franka")
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

import sys
sys.path.append('/home/joel/seonghyeon/World-Model')
from lwm.delta_action_sampler_override_bridge import DeltaActionSampler
from lwm.action_sampler_bridge import ActionSampler
from tux import JaxDistributedConfig
from lwm.delta_llama import VideoLLaMAConfig
from tux import define_flags_with_default, JaxDistributedConfig, set_random_seed
import csv

class FLAGSClass:
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)


class LWMServer:
    def __init__(
            self, 
            load_checkpoint: Union[str, Path], 
            vqgan_checkpoint: Union[str, Path], 
            seed: int,
            mesh_dim: str, 
            dtype: str, 
            load_llama_config: str, 
            updata_llama_config: str, 
            tokens_per_delta: int, 
            tokens_per_action: int, 
            vocab_file: str, 
            multi_image: int, 
            jax_distributed: dict,
            action_scale_file: str,
            img_aug: int,
        ) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        
        # JaxDistributedConfig.initialize(args.jax_distributed)
        set_random_seed(seed)
        tokenizer = VideoLLaMAConfig.get_tokenizer_config()
        llama = VideoLLaMAConfig.get_default_config()
        tokenizer.vocab_file = vocab_file
        kwargs = {
            "vqgan_checkpoint": vqgan_checkpoint,
            "seed": seed,
            "mesh_dim": mesh_dim,
            "dtype": dtype,
            "load_llama_config": load_llama_config,
            "update_llama_config": updata_llama_config,
            "tokens_per_delta": tokens_per_delta,
            "tokens_per_action": tokens_per_action,
            "vocab_file": vocab_file,
            "multi_image": multi_image,
            "jax_distributed": jax_distributed,
            "action_scale_file": action_scale_file,
            "tokenizer": tokenizer,
            "llama": llama,
            "load_checkpoint": load_checkpoint,
            "image_aug": img_aug,
        }
        self.tokens_per_delta = tokens_per_delta
        flags = FLAGSClass(kwargs)

        if kwargs['tokens_per_delta'] > 0:
            self.model = DeltaActionSampler(flags)
        else: 
            self.model = ActionSampler(flags)
        self.load_checkpoint= load_checkpoint

        self.action_scale_list = []
        with open(action_scale_file, 'r') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                # Convert the string values to float and add them to the csv_data list
                self.action_scale_list.append([float(value) for value in row if value.strip()])

        # self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        # if os.path.isdir(self.load_checkpoint):
        #     with open(Path(self.load_checkpoint) / "dataset_statistics.json", "r") as f:
        #         self.vla.norm_stats = json.load(f)
    def get_averaged_values(self, indices):
        averaged_values = []
        for row_idx, idx in enumerate(indices):
            try:
                value1 = self.action_scale_list[row_idx][idx]
                value2 = self.action_scale_list[row_idx][idx + 1]
                average = (value1 + value2) / 2
            except: 
                print("index out of range")
                average = 1
            averaged_values.append(average)
        return averaged_values

    def predict_action(self, image: np.ndarray, instruction: str) -> Union[np.ndarray, str]:
    
        # Parse payload components

        print("@@@",instruction)
        prompts = [{'image': [image], 'question':instruction}]

       
        outputs, vision_output, text_output = self.model(prompts)
        norm_raw_actions = text_output[0]
        print("norm raw actions", norm_raw_actions)
        indices = norm_raw_actions

        action = self.get_averaged_values(indices)

        return action
        

def inverse_discretize_bins(binned_data, min_value, max_value, num_bins=256):
    bin_size = (max_value - min_value) / num_bins
    # Map each bin index to the lower bound of the corresponding value range
    original_data = (int(binned_data) * bin_size) + min_value
    return original_data


def inverse_discretize(action_bins, min_max_lst):
    action_bins = action_bins.replace("    ", " ")
    action_bins = action_bins.replace("   ", " ")
    action_bins = action_bins.replace("  ", " ")
    action_values = action_bins.split(" ")[1:7]
    new_action_values = []
    for i in range(len(action_values)):
        new_action_values.append(
            inverse_discretize_bins(
                action_values[i],
                min_value=min_max_lst[i][0],
                max_value=min_max_lst[i][1],
            )
        )
    return new_action_values


def call_rt(img, msg, model, steps):
    # IP address always changes every time the server is rebooted. Please check the text-generation-webui cli interface

    start = time.time()

    image = Image.fromarray(img)
    # save the img
    # image.save(f"/home/joel/seonghyeon/World-Model/eval_images/{msg}_{steps}.jpg")    
    # Compress the images using JPEG
    # buffer = io.BytesIO()
    # image.save(buffer, format="JPEG", quality=80)
    # jpeg_data = buffer.getvalue()

    # prompt = f"<image>\nWhat action should the robot take to `{msg}`"
    # data = {"images": jpeg_data, "queries": prompt, "answers": ""}

    action = model.predict_action(image, msg)
    
    print(f"PREDICTION: {action}")

    # action = inverse_discretize(response_txt, minmaxlst)
    # action = np.concatenate((action, [float(response_txt[-1])]))

    print(f"Inference time: {time.time() - start} seconds")

    return action


@hydra.main(
    config_path="../../configs/", config_name="eval_lwm_action_real", version_base="1.1"
)
def run_experiment(cfg):
    '''
    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[{cfg.seed}]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    '''
    set_random_seed(cfg.seed)
    #set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    #device = get_device()
    # logger = configure_logger(logdir, cfg.log.format_strings)

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
    # if cfg.aug.camera_resize is not None:
    #     env = ResizeImageWrapper(
    #         env,
    #         size=cfg.aug.camera_resize,
    #         image_keys=[cn for cn in camera_names],
    #     )
    
    data = {
        "obs": [],
        "act": [],
        "rgbd": [],
    }

    env.seed(cfg.seed)
    obs = env.reset()
    jax_distributed = JaxDistributedConfig.get_default_config()

    model = LWMServer(cfg.load_checkpoint, cfg.vqgan_checkpoint, cfg.seed, cfg.mesh_dim, cfg.dtype, cfg.load_llama_config, cfg.update_llama_config, cfg.tokens_per_delta, cfg.tokens_per_action, cfg.vocab_file, cfg.multi_image, jax_distributed, cfg.action_scale_file, cfg.image_aug)
    print(f'finished loading model')
    steps = 0

    done = False
    while not done:

        start = time.time()

        img = obs[camera_names[0]]        
        data["rgbd"].append(img)

        # call openrt api
        act = call_rt(img, cfg.msg, model, steps)

        
        # step
        next_obs, reward, done, info = env.step(act)

        print(
            f"Time {np.around(time.time()-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(act[0],3)}"
        )

        data["act"].append(act)
        data["obs"].append(obs)
        
        image = next_obs["215122255213_rgb"]
        print(f'getting image saved..')
        #cv2.imshow('Real-time video', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if cfg.save_info:
            directory_path = f"{cfg.base_dir}/{cfg.exp_id}/eval/{cfg.folder_name}/{cfg.trial_num}"
            if not os.path.exists(directory_path):
                # Create the directory
                os.makedirs(directory_path)
                print(f"Directory '{directory_path}' created.")
            cv2.imwrite(f"{directory_path}/{steps}.png",cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        obs = next_obs
        steps+=1

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
