import io
import os
import pickle
import time

import hydra
import imageio
import numpy as np
import requests
from PIL import Image

from robot.wrappers.crop_wrapper import CropImageWrapper
from robot.wrappers.resize_wrapper import ResizeImageWrapper
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import configure_logger
from utils.system import get_device, set_gpu_mode
from openrt.scripts.convert_np_to_hdf5 import normalize, unnormalize


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


def call_rt(img, msg, url, minmaxlst):
    # IP address always changes every time the server is rebooted. Please check the text-generation-webui cli interface

    start = time.time()

    image = Image.fromarray(img)
    # Compress the images using JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    jpeg_data = buffer.getvalue()

    prompt = f"<image>\nWhat action should the robot take to `{msg}`"
    data = {"images": jpeg_data, "queries": prompt, "answers": ""}
    data_bytes = pickle.dumps(data)

    response = requests.post(url, data=data_bytes)

    response_txt = response.content.decode("utf-8")
    print(f"PREDICTION: {response_txt}")

    action = inverse_discretize(response_txt, minmaxlst)
    action = np.concatenate((action, [float(response_txt[-1])]))

    print(f"Inference time: {time.time() - start} seconds")

    return action


@hydra.main(
    config_path="../configs/", config_name="eval_openrt_real", version_base="1.1"
)
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
            image_keys=[cn + "_rgb" for cn in camera_names],
            crop_render=True,
        )

    # resize image observations
    if cfg.aug.camera_resize is not None:
        env = ResizeImageWrapper(
            env,
            size=cfg.aug.camera_resize,
            image_keys=[cn + "_rgb" for cn in camera_names],
        )

    stats = pickle.load(open(os.path.join(cfg.data_path, "stats"), "rb"))

    data = {
        "obs": [],
        "act": [],
        "rgbd": [],
    }

    env.seed(cfg.seed)
    obs = env.reset()

    done = False
    while not done:

        start = time.time()

        img = obs[camera_names[0]]        
        data["rgbd"].append(img)

        # call openrt api
        act = call_rt(img, cfg.msg, cfg.url, cfg.minmaxlst)[None]

        # unnorm action
        act = unnormalize(act, stats["action"])
        
        # step
        next_obs, reward, done, info = env.step(act)

        print(
            f"Time {np.around(time.time()-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(act[0],3)}"
        )

        data["act"].append(act)
        data["obs"].append(obs)
        obs = next_obs

    # stack acts and imgs
    for k in ["act", "rgbd"]:
        data[k] = np.stack(data[k])

    data_obs_tmp = {}
    for k in data["obs"][0].keys():
        data_obs_tmp[k] = data["obs"][0][k][None]

    for obs in data["obs"][1:]:
        for k, v in obs.items():
            data_obs_tmp[k] = np.concatenate((data_obs_tmp[k], v[None]))

    imageio.mimwrite(
        os.path.join(logdir, "open_rt_eval.gif"), data["rgbd"].squeeze(), duration=10
    )


if __name__ == "__main__":
    run_experiment()
