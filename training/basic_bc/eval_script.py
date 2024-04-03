import collections
import os
from dataclasses import dataclass

import gym
import hydra
import imageio

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import numpy as np
import torch
from tqdm.auto import tqdm

from robot.crop_wrapper import CropImageWrapper
from robot.resize_wrapper import ResizeImageWrapper
from robot.sim.vec_env.vec_env import make_env
from training.basic_bc.models.policies import MixedGaussianPolicy
from training.basic_bc.train_script import plot_trajectory
from training.weird_diffusion.datasets.utils import (normalize_data,
                                                     unnormalize_data)
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Image, Video, configure_logger
from utils.system import get_device, set_gpu_mode


def pre_process_obs(obs, cfg, stats, device):
    state = None
    if len(cfg.training.state_keys):
        sts = []
        for key in cfg.training.state_keys:
            st = obs[key]
            sts.append(st)
        state = np.concatenate(sts, axis=0)
        # normalize
        state = normalize_data(state, stats["state"])
        # to tensor to gpu
        state = torch.tensor(state[None], dtype=torch.float32).to(device)

    image = None
    if len(cfg.training.image_keys):
        imgs = []
        for key in cfg.training.image_keys:
            assert key in obs.keys(), f"Key {key} not in obs.keys() {obs.keys()}!"
            img = obs[key]
            img = np.moveaxis(img, -1, 0)
            imgs.append(img)
        image = np.stack(imgs, axis=0)
        # normalize
        img_min = stats["image"]["min"].mean()
        img_max = stats["image"]["max"].mean()
        image = (image - img_min) / (img_max - img_min)
        image = image * 2 - 1
        # to tensor to gpu
        image = torch.tensor(image[None], dtype=torch.float32).to(device)

    return image, state


@hydra.main(version_base=None, config_path="../../configs", config_name="bc_policy_sim")
def run_experiment(cfg):
    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[{cfg.seed}]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    set_random_seed(cfg.seed)

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed))
    logger = configure_logger(logdir, cfg.log.format_strings)

    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    checkpoint = torch.load(os.path.join(logdir, "bc_policy"), map_location="cuda")
    stats = checkpoint["stats"]

    policy = MixedGaussianPolicy(
        img_shape=(
            stats["image"]["max"].shape[1:] if len(cfg.training.image_keys) else None
        ),
        state_shape=(
            stats["state"]["max"].shape if len(cfg.training.state_keys) else None
        ),
        act_shape=stats["action"]["max"].shape,
        hidden_dim=cfg.training.hidden_dim,
    ).to(device)

    policy.load_state_dict(checkpoint["state_dict"])
    print("Pretrained weights loaded.")

    cfg.robot.DoF = 6
    cfg.robot.control_hz = 10 if cfg.robot.ip_address is None else 1
    cfg.robot.gripper = True
    cfg.robot.blocking_control = True
    cfg.robot.on_screen_rendering = False
    cfg.robot.max_path_length = 100

    cfg.env.flatten = False
    # cfg.env.obj_pose_noise_dict = None

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )
    camera_names = env.unwrapped._robot.camera_names.copy()
    env.action_space.low[:3] = -0.1
    env.action_space.high[:3] = 0.1
    env.action_space.low[3:] = -0.25
    env.action_space.high[3:] = 0.25

    env = CropImageWrapper(
        env,
        y_min=80,
        y_max=-80,
        image_keys=[camera_names[0] + "_rgb"],
        crop_render=True,
    )
    env = ResizeImageWrapper(
        env, size=(224, 224), image_keys=[camera_names[0] + "_rgb"]
    )

    obs = env.reset()
    # np.save("eval_obs", obs)

    done = False
    imgs = []
    acts = []
    while not done:
        img, state = pre_process_obs(obs, cfg, stats, device)
        # np.save("process_obs", {"img": img.cpu().detach().numpy(), "state": state.cpu().detach().numpy()})
        # imgs.append(img[0].detach().cpu().numpy())
        imgs.append(env.render())
        with torch.no_grad():
            act = (
                policy.forward(img, state, deterministic=False)[0].detach().cpu().numpy()
            )
            act = unnormalize_data(act, stats["action"])
            act[-1] = 0 if act[-1] < 0.7 else 1
        acts.append(act)
        obs, reward, done, _ = env.step(act)

    plot_img = plot_trajectory(
        pred_actions=np.stack(acts),
        true_actions=None,
        imgs=np.stack(imgs).transpose(0, 3, 1, 2),
    )
    logger.record(
        f"images/eval",
        Image(plot_img, dataformats="HWC"),
        exclude=["stdout"],
    )

    video = np.stack(imgs)
    imageio.mimsave(os.path.join(logdir, f"eval.mp4"), video)
    logger.record(
        f"videos/eval",
        Video(video, fps=20),
        exclude=["stdout"],
    )

    logger.dump()


if __name__ == "__main__":
    run_experiment()
