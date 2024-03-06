import os
import time

import hydra
import joblib
import torch

from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Video, configure_logger
from utils.system import get_device, set_gpu_mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np

from asid.wrapper.asid_vec import make_env, make_vec_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.pointclouds import *


def viz_points(points):
    """quick visualization: viz_points(env.get_points())"""
    points = np.concatenate((points[0, 0], points[0, 1]), axis=0)
    points = crop_points(points)
    visualize_pcds([points_to_pcd(points)])


@hydra.main(
    config_path="../configs/", config_name="explore_rod_real", version_base="1.1"
)
def run_experiment(cfg):

    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[explore][{cfg.seed}]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    set_random_seed(cfg.seed)
    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "explore")
    logger = configure_logger(logdir, cfg.log.format_strings)

    cfg.num_workers = 1
    cfg.asid.obs_noise = 0.0

    # real env
    envs = make_vec_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid) if cfg.robot.ip_address is None else None,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        device_id=0,
        verbose=False,
    )

    ckptdir = os.path.join(
        # logdir, "policy_step_9001" # custom ckpt
        logdir,
        "policy",
    )

    from stable_baselines3 import SAC

    policy = SAC("MlpPolicy", envs, device=device)
    policy = policy.load(ckptdir)

    data = {
        "obs": [],
        "act": [],
        "rgbd": [],
    }

    # if cfg.env.sim and (hasattr(cfg.train, "ood_params") and cfg.train.ood_params):
    #     param_dim = env.get_parameters()[0].shape[0]
    #     # sample params in (normalized) range [-1.6, -1.1] or [1.1, 1.6]
    #     rnd = np.random.uniform(low=-0.5, high=0.5, size=(1, param_dim))
    #     param_ood = rnd + np.sign(rnd) * 1.1
    #     env.set_parameters(param_ood)

    envs.seed(cfg.seed)
    obs = envs.reset()

    done = False
    while not done:

        images_array = envs.render()
        data["rgbd"].append(images_array)

        act, _ = policy.predict(obs, deterministic=False)
        next_obs, reward, done, info = envs.step(act)

        print(
            f"EE {np.around(obs[0,:2],3)} Obj {np.around(obs[0,11:13],3)} Act {np.around(act,3)}"
        )

        data["act"].append(act)
        data["obs"].append(obs)
        obs = next_obs

    if cfg.robot.ip_address is None:
        data["zeta"] = np.array(envs.get_parameters()[0])

    for k, v in data.items():
        data[k] = np.stack(v)

    import imageio

    imageio.mimwrite("explore.gif", data["rgbd"].squeeze(), duration=10)
    # b, t, c, h, w
    video = np.transpose(data["rgbd"][..., :3], (1, 0, 4, 2, 3))
    logger.record(
        f"eval_policy/traj",
        Video(video, fps=10),
        exclude=["stdout"],
    )
    logger.dump(step=0)
    filenames = joblib.dump(data, os.path.join(logdir, f"rollout.pkl"))
    print("Saved rollout to", filenames)


if __name__ == "__main__":
    run_experiment()
