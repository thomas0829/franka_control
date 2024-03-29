import collections
import os
from dataclasses import dataclass

import gym
import hydra
import imageio
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import numpy as np
import torch
from tqdm.auto import tqdm

from robot.sim.vec_env.vec_env import make_env
from training.weird_diffusion.datasets.utils import normalize_data
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Video, configure_logger
from utils.system import get_device, set_gpu_mode

from training.basic_bc.models.policies import MixedGaussianPolicy


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


@hydra.main(
    version_base=None, config_path="../../configs", config_name="bc_policy_sim"
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

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed))
    logger = configure_logger(logdir, cfg.log.format_strings)

    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    checkpoint = torch.load(os.path.join(logdir, "bc_policy"), map_location="cuda")
    stats = checkpoint["stats"]

    policy = MixedGaussianPolicy(
        img_shape=stats["image"]["max"].shape[1:] if len(cfg.training.image_keys) else None,
        state_shape=stats["state"]["max"].shape if len(cfg.training.state_keys) else None,
        act_shape=stats["action"]["max"].shape,
        hidden_dim=128,
    ).to(device)

    policy.load_state_dict(checkpoint["state_dict"])
    print("Pretrained weights loaded.")

    cfg.robot.max_path_length = cfg.inference.max_steps
    cfg.robot.blocking_control = True

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env) if cfg.robot.ip_address is None else None,
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    from robot.crop_wrapper import CropImageWrapper
    env = CropImageWrapper(env, y_min=160, image_keys=cfg.training.image_keys)

    obs = env.reset()

    done = False
    imgs = []

    while not done:
        img, state = pre_process_obs(obs, cfg, stats, device)
        with torch.no_grad():
            print(img, state)
            act = policy.forward(img, state)[0].detach().cpu().numpy()
        print("action", act)
        obs, reward, done, _ = env.step(act)
        imgs.append(env.render())

    video = np.stack(imgs)[None]
    imageio.mimsave(os.path.join(logdir, f"eval.mp4"), video[0])
    logger.record(
        f"videos/eval",
        Video(video, fps=20),
        exclude=["stdout"],
    )


if __name__ == "__main__":
    run_experiment()
