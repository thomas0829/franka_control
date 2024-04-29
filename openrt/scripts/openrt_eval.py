import os
import base64
import requests
import time
import hydra
import joblib
import imageio

from utils.experiment import (
    setup_wandb,
    hydra_to_dict,
    set_random_seed,
)
from utils.system import set_gpu_mode, get_device
from utils.logger import configure_logger, Video

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils.experiment import (
    setup_wandb,
    hydra_to_dict,
    set_random_seed,
)

import numpy as np
from asid.wrapper.asid_vec import make_vec_env

from utils.pointclouds import *
import io
import pickle
from PIL import Image

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


# def call_rt(msg):
#     ip_address = 'https://contributions-provides-bound-spanking.trycloudflare.com'
#     # IP address always changes every time the server is rebooted. Please check the text-generation-webui cli interface
#     start = time.time()
#     with open('tmp.png', 'rb') as f:
#         img_str = base64.b64encode(f.read()).decode('utf-8')
#         prompt = f'### human: \nWhat action should the robot take to `{msg}`\n<img src="data:image/jpeg;base64,{img_str}">### gpt: '
#         response = requests.post(f'{ip_address}/v1/completions', json={'prompt': prompt, 'max_tokens': 256, 'stopping_strings': ['\n###']}).json()
#         reponse_txt = response['choices'][0]['text']
#         minmaxlst = [[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]]

#         action = inverse_discretize(reponse_txt, minmaxlst)[:3]
#         action = np.concatenate((action, np.ones(1)))

#     end = time.time()
#     print(f'Took { end - start } seconds')
#     return action


def call_rt(img, msg, url, minmaxlst):
    # IP address always changes every time the server is rebooted. Please check the text-generation-webui cli interface

    start = time.time()

    image = Image.fromarray(img)
    # Compress the images using JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    jpeg_data = buffer.getvalue()

    prompt = f'<image>\nWhat action should the robot take to `{msg}`'
    data = {"images": jpeg_data, "queries": prompt, "answers": ""}
    data_bytes = pickle.dumps(data)
    
    response = requests.post(url, data=data_bytes)
    
    response_txt = response.content.decode("utf-8")
    print(f'PREDICTION: {response_txt}')
    
    action = inverse_discretize(response_txt, minmaxlst)
    action = np.concatenate((action, [float(response_txt[-1])]))

    print(f"Inference time: {time.time() - start} seconds")

    return action


@hydra.main(config_path="../configs/", config_name="openrt_eval", version_base="1.1")
def run_experiment(cfg):

    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[{cfg.seed}]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    set_random_seed(cfg.seed)
    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed))
    logger = configure_logger(logdir, cfg.log.format_strings)

    cfg.robot.control_hz = 1
    cfg.robot.blocking_control = True

    # real env
    # TODO move to make_env
    envs = make_vec_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        device_id=0,
        verbose=False,
    )

    data = {
        "obs": [],
        "act": [],
        "rgbd": [],
    }

    envs.seed(cfg.seed)
    obs = envs.reset()

    done = False
    while not done:

        start = time.time()

        # preprocess img
        img = obs[0]["215122255213_rgb"][:, 160:]
        data["rgbd"].append(img)

        # call openrt api
        act = call_rt(img, cfg.inference.msg, cfg.inference.url, cfg.inference.minmaxlst)[None]
        # act = envs.action_space.sample()[None]

        # step
        next_obs, reward, done, info = envs.step(act)

        print(
            f"Time {np.around(time.time()-start, 3)} EE {np.around(obs[0]['lowdim_ee'][:3],3)} Act {np.around(act[0],3)}"
        )

        data["act"].append(act)
        data["obs"].append(obs)
        obs = next_obs

    # stack acts and imgs
    for k in ["act", "rgbd"]:
        data[k] = np.stack(data[k])

    data_obs_tmp = {}
    for k in data["obs"][0][0].keys():
        data_obs_tmp[k] = data["obs"][0][0][k][None]

    for obs in data["obs"][1:]:
        for k, v in obs[0].items():
            data_obs_tmp[k] = np.concatenate((data_obs_tmp[k], v[None]))

    imageio.mimwrite("open_rt_eval.gif", data["rgbd"].squeeze(), duration=10)

    # # b, t, c, h, w
    # video = np.transpose(data["rgbd"][..., :3], (1, 0, 4, 2, 3))
    # logger.record(
    #     f"eval_policy/traj",
    #     Video(video, fps=10),
    #     exclude=["stdout"],
    # )
    # logger.dump(step=0)

    # # joblib.dump(data, os.path.join(logdir, f"rollout_{'sim' if cfg.env.sim else 'real'}.pkl"))
    # # for rod that doesn're require reconstruction
    # filenames = joblib.dump(data, os.path.join(logdir, f"rollout.pkl"))
    # print("Saved rollout to", filenames)
    # # print(os.path.join(logdir, f"rollout_{'sim' if cfg.env.sim else 'real'}.pkl"))


if __name__ == "__main__":
    run_experiment()
