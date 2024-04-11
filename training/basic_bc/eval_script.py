import collections
import os
from dataclasses import dataclass

import cv2
import gym
import hydra
import imageio

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import numpy as np
import torch
from tqdm import trange

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
        # # normalize -> done in model now!
        # img_min = stats["image"]["min"].mean()
        # img_max = stats["image"]["max"].mean()
        # image = (image - img_min) / (img_max - img_min)
        # image = image * 2 - 1
        # # to tensor to gpu
        image = torch.tensor(image[None], dtype=torch.float32).to(device)

    return image, state


def is_success(obj_poses):
    return obj_poses[:, 2] > 0.1


@hydra.main(version_base=None, config_path="../../configs", config_name="bc_policy_sim")
def run_experiment(cfg):
    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[{cfg.seed}][eval]",
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
        img_shape=stats["image"]["max"].shape if len(cfg.training.image_keys) else None,
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
    # fake_blocking = cfg.robot.blocking_control
    # cfg.robot.blocking_control = False
    fake_blocking = False
    cfg.robot.blocking_control = True
    cfg.robot.on_screen_rendering = False
    cfg.robot.max_path_length = 100
    cfg.env.flatten = False
    cfg.robot.imgs = True
    cfg.robot.calibration_file = None

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
        image_keys=[cn + "_rgb" for cn in camera_names],
        crop_render=True,
    )
    env = ResizeImageWrapper(
        env, size=(224, 224), image_keys=[cn + "_rgb" for cn in camera_names]
    )

    obj_poses = []
    
    for i in trange(100):

        obs = env.reset()

        done = False
        imgs = []
        acts = []

        while not done:

            img, state = pre_process_obs(obs, cfg, stats, device)

            # save image and resize
            if len(cfg.training.image_keys):
                # imgs.append(img[0].detach().cpu().numpy())
                img_tmp = obs[cfg.training.image_keys[0]]
            else:
                img_tmp = env.render()
            img_resize = cv2.resize(img_tmp, dsize=(128, 128))
            imgs.append(img_resize.transpose(2, 0, 1)[None])

            with torch.no_grad():
                act = (
                    policy.forward(img, state, deterministic=True)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                acts.append(act)
                act = unnormalize_data(act, stats["action"])
                act[-1] = 1.0 if act[-1] > 0.5 else 0.0
            obs, reward, done, _ = env.step(act)

        # obj poses for success check
        obj_poses.append(obs["obj_pose"])

        # T,C,H,W
        video = np.stack(imgs)[:, 0]

        # save trajectory plot -> takes T,C,H,W
        plot_img = plot_trajectory(
            pred_actions=np.stack(acts),
            true_actions=None,
            imgs=video,
        )
        logger.record(
            f"images/eval_{i}",
            Image(plot_img, dataformats="HWC"),
            exclude=["stdout"],
        )

        # save video local -> takes T,H,W,C
        imageio.mimsave(
            os.path.join(logdir, f"eval_{i}.mp4"), video.transpose(0, 2, 3, 1)
        )
        # save video wandb -> takes T,C,H,W
        logger.record(
            f"videos/eval_{i}",
            Video(video, fps=20),
            exclude=["stdout"],
        )

    logger.record("success", np.mean(is_success(np.stack(obj_poses))))

    logger.dump()


if __name__ == "__main__":
    run_experiment()
