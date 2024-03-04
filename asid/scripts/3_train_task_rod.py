import argparse
import os
from multiprocessing import Process, Queue

import hydra
import imageio
import joblib
import mujoco
import numpy as np
import torch
from tqdm import trange

from asid.wrapper.asid_vec import make_env, make_vec_env

from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Video, configure_logger
from utils.system import get_device, set_gpu_mode
from utils.transformations import *
from utils.transformations_mujoco import *

from asid.utils.move import collect_rollout


def train_cem_policy(cfg, zeta=None):

    num_iters, num_samples, num_procs = (
        cfg.train.algorithm.num_iters,
        cfg.train.algorithm.num_samples,
        cfg.train.algorithm.num_workers,
    )
    num_elites = int(cfg.train.algorithm.elite_frac * cfg.train.algorithm.num_samples)

    action_mean = cfg.train.algorithm.mu_init
    action_std = cfg.train.algorithm.sigma_init

    q = Queue()
    batch_size = num_samples // num_procs
    for _ in trange(num_iters, desc="CEM iteration"):

        procs = [
            Process(
                target=cem_rollout_worker,
                args=(q, cfg, batch_size, action_mean, action_std, zeta),
            )
            for _ in range(num_procs)
        ]

        for p in procs:
            p.start()

        for p in procs:
            p.join()

        # Update mean and std
        results = [q.get() for _ in range(num_procs)]
        actions = np.concatenate([res["actions"] for res in results], axis=0)
        rewards = np.concatenate([res["rewards"] for res in results], axis=0)

        elites = actions[np.cfgort(rewards)][-num_elites:]
        action_mean = np.mean(elites, axis=0)
        action_std = np.std(elites, axis=0)
        print(
            f"action_mean: {action_mean}, action_std: {action_std}, reward: {np.mean(rewards)} zeta: {zeta}"
        )
    return action_mean, action_std


def cem_rollout_worker(
    q,
    cfg,
    num_rollouts,
    action_mean,
    action_std,
    zeta,
    verbose=False,
    render=False,
):

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed,
        device_id=cfg.gpu_id,
        collision=True,
    )

    actions = np.zeros((num_rollouts), dtype=np.float32)
    rewards = np.zeros((num_rollouts), dtype=np.float32)

    for i in range(num_rollouts):

        if zeta is not None:
            env.set_parameters(zeta)

        # Sample action
        action = np.random.normal(action_mean, action_std)

        # Collect rollout -> resets env
        reward, _ = collect_rollout(
            env,
            action,
            control_hz=cfg.robot.control_hz,
            verbose=verbose,
            render=render,
        )

        actions[i] = action
        rewards[i] = reward

    return actions, rewards


@hydra.main(config_path="../configs/", config_name="task_rod_sim", version_base="1.1")
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

    # cfg.robot.on_screen_rendering = True
    cfg.robot.gripper = True
    cfg.asid.obs_noise = 0.0

    # Load zeta parameter
    zeta_dir = os.path.join(logdir, cfg.exp_id, str(cfg.seed), "sysid", "zeta")
    if os.path.exists(zeta_dir):
        zeta_dict = joblib.load(zeta_dir)
        for k, v in zeta_dict.items():
            zeta_dict[k] = np.array(v)
    else:
        print("Using default zeta_dict")
        zeta_dict = {"mu": np.array([0.07]), "": np.array([0.08])}

    # Train policy
    if cfg.train.mode == "manual":
        action = cfg.train.action

    else:
        if cfg.train.mode == "sysid":
            action_mean, action_std = train_cem_policy(cfg, zeta_dict["mu"])
        elif cfg.train.mode == "domain_rand":
            action_mean, action_std = train_cem_policy(cfg, None)

        action = np.random.normal(action_mean, action_std)
        print(
            f"{cfg.exp_id} {cfg.train.mode} action_mean: {action_mean}, action_std: {action_std}"
        )

        param_dict = {
            "mu": action_mean,
            "sigma": action_std,
        }
        joblib.dump(
            param_dict,
            os.path.join(logdir, cfg.exp_id, str(cfg.seed), "task", "policy"),
        )

    # Evaluate policy
    eval_env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed + 100,
        device_id=cfg.gpu_id,
        collision=True,
    )
    eval_env.set_parameters(zeta_dict[""])
    video_path = os.path.join(cfg.logdir, cfg.exp_id, str(cfg.seed), "sysid")
    os.makedirs(video_path, exist_ok=True)
    reward = collect_rollout(
        eval_env,
        action,
        log_video=True,
        video_path=os.path.join(video_path, f"{cfg.train.mode}.gif"),
    )
    print(
        f"FINAL real zeta {zeta_dict['']} EXP {cfg.exp_id} ALGO {cfg.train.mode} reward: {reward} act {action} act_mean {action_mean} act_std {action_std}"
    )


if __name__ == "__main__":
    run_experiment()
