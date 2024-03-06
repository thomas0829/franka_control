import argparse
import os
from multiprocessing import Process, Queue
from multiprocessing import Pool

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


def train_cem_policy(cfg, zeta=None, obj_pose=None):

    num_iters, num_samples, num_procs = (
        cfg.train.algorithm.num_iters,
        cfg.train.algorithm.num_samples,
        cfg.train.algorithm.num_workers,
    )
    num_elites = int(cfg.train.algorithm.elite_frac * cfg.train.algorithm.num_samples)

    action_mean = cfg.train.algorithm.mu_init
    action_std = cfg.train.algorithm.sigma_init

    batch_size = num_samples // num_procs
    batch_size = 1
    for _ in trange(num_iters, desc="CEM iteration"):
        # q = Queue()
        # with Pool(num_procs) as p:
        #     results = p.map(
        #         cem_rollout_worker,
        #         [
        #             (cfg, batch_size, action_mean, action_std, zeta, cfg.seed + i)
        #             for i in range(num_procs)
        #         ],
        #     )

        # procs = [
        #     Process(target=cem_rollout_worker, args=(q, cfg, batch_size, action_mean, action_std, zeta))
        #     for _ in range(num_procs)
        # ]
        
        # for p in procs:
        #     p.start()

        # for p in procs:
        #     p.join()

        # # Update mean and std
        # results = [q.get() for _ in range(num_procs)]
        # actions = np.concatenate([res["actions"] for res in results], axis=0)
        # rewards = np.concatenate([res["rewards"] for res in results], axis=0)

        # elites = actions[np.argsort(rewards)][-num_elites:]
        # action_mean = np.mean(elites, axis=0)
        # action_std = np.std(elites, axis=0)
        # print(f"action_mean: {action_mean}, action_std: {action_std}, reward: {np.mean(rewards)} zeta: {zeta}")

        results = []
        for i in trange(num_samples, desc="collecting rollouts..."):
            res = cem_rollout_worker(cfg, batch_size, action_mean, action_std, zeta, obj_pose, cfg.seed + i)
            results.append(res)

        # Update mean and std
        actions = np.concatenate([res["act"] for res in results], axis=0)
        rewards = np.concatenate([res["rew"] for res in results], axis=0)

        elites = actions[np.argsort(rewards)][-num_elites:]
        action_mean = np.mean(elites, axis=0)
        action_std = np.std(elites, axis=0)
        print(
            f"action_mean: {action_mean}, action_std: {action_std}, reward: {np.mean(rewards)} zeta: {zeta}"
        )
    return action_mean, action_std


def cem_rollout_worker(
    # q,
    cfg,
    num_rollouts,
    action_mean,
    action_std,
    zeta,
    obj_pose,
    seed=0,
    verbose=False,
    render=False,
):
    
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed,
        device_id=cfg.gpu_id,
        # TODO
        collision=False,
    )

    actions = np.zeros((num_rollouts), dtype=np.float32)
    rewards = np.zeros((num_rollouts), dtype=np.float32)

    for i in range(num_rollouts):
        
        if zeta is not None:
            env.set_parameters(zeta)
        if obj_pose is not None:
            env.set_obj_pose(obj_pose)

        # Sample action
        np.random.seed(seed)
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

    return {"act": actions, "rew": rewards}
    # q.put({"actions": actions, "rewards": rewards})


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
    cfg.robot.DoF = 6
    cfg.robot.gripper = True
    cfg.robot.max_path_length = 1e5
    cfg.robot.on_screen_rendering = False

    cfg.env.obs_keys = ["lowdim_ee", "lowdim_qpos"]
    cfg.env.obj_pos_noise = False

    cfg.asid.obs_noise = 0.0
    cfg.asid.reward = False

    # Load zeta parameter
    zeta_dir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "sysid", "zeta")
    if os.path.exists(zeta_dir):
        sysid_dict = joblib.load(zeta_dir)
        for k, v in sysid_dict.items():
            sysid_dict[k] = np.array(v)
        zeta = sysid_dict["mu"]
        obj_pose = sysid_dict["final_obs"][-7:]
    else:
        zeta = np.array([0.07])
        obj_pose = np.array([0.4, 0.3, 0.02, 0, 0, 0, 0])

    # Train policy
    if cfg.train.mode == "manual":
        action = cfg.train.action
    else:
        if cfg.train.mode == "sysid":
            action_mean, action_std = train_cem_policy(cfg, zeta=zeta, obj_pose=obj_pose)
        elif cfg.train.mode == "domain_rand":
            action_mean, action_std = train_cem_policy(cfg, zeta=None, obj_pose=obj_pose)

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
    eval_env.set_parameters(sysid_dict[""])
    video_path = os.path.join(cfg.logdir, cfg.exp_id, str(cfg.seed), "sysid")
    os.makedirs(video_path, exist_ok=True)
    reward = collect_rollout(
        eval_env,
        action,
        log_video=True,
        video_path=os.path.join(video_path, f"{cfg.train.mode}.gif"),
    )
    print(
        f"FINAL real zeta {sysid_dict['']} EXP {cfg.exp_id} ALGO {cfg.train.mode} reward: {reward} act {action} act_mean {action_mean} act_std {action_std}"
    )


if __name__ == "__main__":
    run_experiment()
