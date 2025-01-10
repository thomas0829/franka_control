import io
import os
import pickle
import time

import hydra
import imageio
import numpy as np
import requests
from PIL import Image

import sys
sys.path.append("/home/joel/projects/robomimic_openrt")
sys.path.append("/home/joel/projects/robosuite")
sys.path.append("/home/joel/projects/polymetis_franka")
from robot.wrappers.crop_wrapper import CropImageWrapper
from robot.wrappers.resize_wrapper import ResizeImageWrapper
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import configure_logger
from utils.system import get_device, set_gpu_mode

import cv2

import json
import base64

def decode_numpy(json_obj):
    if "__numpy__" in json_obj:
        data = base64.b64decode(json_obj["__numpy__"])
        dtype = np.dtype(json_obj["dtype"])
        shape = tuple(json_obj["shape"])
        array = np.frombuffer(data, dtype=dtype).reshape(shape)
        return array
    return json_obj

def call_rt(img, msg, url):
    # IP address always changes every time the server is rebooted. Please check the text-generation-webui cli interface
    start = time.time()
    prompt = msg
    data = {"image": img.tolist() if isinstance(img, np.ndarray) else img, "instruction": prompt}
    payload_json = json.dumps(data)
    headers = {
        'Content-Type': 'application/json',
    }
    response = requests.post(url, data=payload_json, headers=headers)
    decoded_json = json.loads(response.content)
    action = decode_numpy(decoded_json)
    print(f'Action: {action}')
    print(f"Inference time: {time.time() - start} seconds")

    return action


@hydra.main(
    config_path="../../configs/", config_name="eval_lwm_latent_real", version_base="1.1"
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
    # set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    # device = get_device()

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

    trial_num = 0
    data = {
        "obs": [],
        "act": [],
        "rgbd": [],
    }

    env.seed(cfg.seed)
    obs = env.reset()

    done = False
    count = 0
    
    for i in range(cfg.traj_length):
        print(f'Step: {i}')
        start = time.time()

        img = obs[camera_names[0]]        
        data["rgbd"].append(img)

        # call openrt api
        act = call_rt(img, cfg.msg, cfg.url)
        
        # step
        next_obs, reward, done, info = env.step(act)

        print(
            f"Time {np.around(time.time()-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(act[0],3)}"
        )

        data["act"].append(act)
        data["obs"].append(obs)

        image = next_obs["215122255213_rgb"]

        from datetime import date
        todays_date = date.today() 
        # cv2.imshow('Real-time video', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if cfg.save_info:
            if cfg.latent:
                model_name = "lwm_latent"
            else:
                model_name =  "lwm_nolatent"
            directory_path = f"{cfg.base_dir}/{todays_date.month}{todays_date.day}/{cfg.exp_id}_eval/{model_name}/{cfg.trial_num}"
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
