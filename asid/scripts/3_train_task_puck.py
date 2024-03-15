import argparse
import os
from multiprocessing import Process, Queue
from multiprocessing import Pool

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

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

# from asid.utils.move import collect_rollout


# move to strike pose
# def move_to_puck(env, error_threshold=1e-2, max_steps=100, render=False):

#     imgs = []
#     target_pose = env.get_obj_pose()[:2]
#     target_pose[0] -= 0.04
#     curr_pose = env.unwrapped._robot.get_ee_pose()[:2]
#     error = np.linalg.norm(target_pose - curr_pose)

#     steps = 0
#     while error > error_threshold and steps < max_steps:
#         steps += 1
#         print(steps, error)
#         act = target_pose - curr_pose
#         env.step(act)
#         curr_pose = env.unwrapped._robot.get_ee_pose()[:2]
#         error = np.linalg.norm(target_pose - curr_pose)
#         if render:
#             imgs.append(env.render())

#     return imgs


from asid.utils.puck import pre_reset_env_mod, post_reset_env_mod, move_to_puck, collect_rollout


# def collect_rollout(env, cfg, action, goal_x, verbose=False, render=False):

#     pre_reset_env_mod(env, cfg)
#     env.reset()
#     post_reset_env_mod(env, cfg)

#     imgs = []
#     # move to strike pos
#     tmp = move_to_puck(env, error_threshold=1e-2)
#     imgs.extend(tmp)

#     # strike
#     for i in range(1):
#         env.step(np.array([action, 0.0]))
#         if render:
#             tmp = [env.render()]
#             imgs.extend(tmp)

#     # step sim until puck stops moving
#     for i in range(20):
#         env.step(np.array([-0.01, 0.0]))
#         if render:
#             tmp = [env.render()]
#             imgs.extend(tmp)

#     # compute reward
#     obj_pose = env.get_obj_pose()
#     reward = -np.linalg.norm(obj_pose[0] - goal_x)

#     return reward, imgs


def train_cem_policy(cfg, zeta=None, obj_pose=None, goal_x=None, render=False):

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

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed,
        device_id=cfg.gpu_id,
        collision=False,
    )

    for j in trange(num_iters, desc="CEM iteration"):

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
            res = cem_rollout_worker(
                env,
                cfg,
                batch_size,
                action_mean,
                action_std,
                zeta,
                obj_pose,
                goal_x,
                False,
                render and i == 0,
            )
            results.append(res)

            if i == 0 and not cfg.robot.on_screen_rendering:
                imgs = np.concatenate([res["imgs"] for res in results], axis=0)
                imageio.mimsave(
                    os.path.join(
                        cfg.log.dir, cfg.exp_id, str(cfg.seed), "task", f"cem_{j}.mp4"
                    ),
                    imgs,
                )

        # Update mean and std
        actions = np.concatenate([res["act"] for res in results], axis=0)
        rewards = np.concatenate([res["rew"] for res in results], axis=0)

        elites = actions[np.argsort(rewards)][-num_elites:]
        action_mean = np.mean(elites, axis=0)
        action_std = np.std(elites, axis=0)
        print(
            f"action_mean: {action_mean}, action_std: {action_std}, reward: {np.mean(rewards)} zeta: {zeta}, best action: {actions[np.argsort(rewards)][-1]}, best reward: {rewards[np.argsort(rewards)][-1]}"
        )
        # TODO put back into place
        if action_std < 3e-3:
            break
    return action_mean, action_std


def cem_rollout_worker(
    # q,
    env,
    cfg,
    num_rollouts,
    action_mean,
    action_std,
    zeta,
    obj_pose,
    goal_x,
    verbose=False,
    render=False,
):

    actions = np.zeros((num_rollouts), dtype=np.float32)
    rewards = np.zeros((num_rollouts), dtype=np.float32)

    for i in range(num_rollouts):

        if zeta is not None:
            env.set_parameters(zeta)
        if obj_pose is not None:
            env.set_obj_pose(obj_pose)

        # Sample action
        # np.random.seed(seed)
        action = np.random.normal(action_mean, action_std)

        # Collect rollout -> resets env
        reward, imgs = collect_rollout(
            env,
            cfg,
            action,
            goal_x,
            verbose=verbose,
            render=render and i == 0,
        )
        actions[i] = action
        rewards[i] = reward
        print(f"Rollout {i} reward: {reward} action: {action}")
    return {"act": actions, "rew": rewards, "imgs": imgs}
    # q.put({"actions": actions, "rewards": rewards})


@hydra.main(config_path="../configs/", config_name="task_puck_sim", version_base="1.1")
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

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "task")
    logger = configure_logger(logdir, cfg.log.format_strings)

    # cfg.robot.on_screen_rendering = True
    cfg.robot.DoF = 2
    cfg.robot.gripper = False 
    cfg.robot.max_path_length = 1e5
    cfg.robot.on_screen_rendering = False

    cfg.env.obs_keys = ["lowdim_ee", "lowdim_qpos"]
    
    cfg.robot.control_hz = 10

    # debug sim
    # cfg.asid.parameter_dict = {}

    cfg.asid.obs_noise = 0.0
    cfg.asid.reward = False

    # goals
    board_x = 0.67
    goal_x = board_x + 0.205
    # goal_x = board_x + 0.455

    # Load zeta parameter
    zeta_dir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "sysid", "zeta")
    if os.path.exists(zeta_dir):
        sysid_dict = joblib.load(zeta_dir)
        for k, v in sysid_dict.items():
            sysid_dict[k] = np.array(v)
        zeta = sysid_dict["mu"]
        # obj_pose = sysid_dict["final_obs"][-7:]

        # obj_pose[2] = 0.0427
        obj_pose = None
        rollout = joblib.load(
        os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "explore", "rollout.pkl")
    )

    else:
        zeta = None
        obj_pose = None

    explore_dir = os.path.join(
        cfg.log.dir, cfg.exp_id, str(cfg.seed), "explore", "rollout.pkl"
    )
    if os.path.exists(explore_dir):
        explore_dict = joblib.load(explore_dir)
        # rod flipped -> apply inverse zeta
        if cfg.env.obj_id == "rod" and  np.abs(quat_to_euler_mujoco(sysid_dict["final_obs"][-4:])[-1]) > np.pi / 4:
            zeta = -sysid_dict["mu"]
    # else:
    #     zeta = np.array([0.02949091])
    #     obj_pose = np.array([0.4, 0.3, 0.02, 0, 0, 0, 0])

    # Train policy
    if cfg.train.mode == "manual":
        action = cfg.train.action
    else:
        if cfg.train.mode == "sysid":
            cfg.env.obj_pose_noise_dict = None
            action_mean, action_std = train_cem_policy(
                cfg, zeta=zeta, obj_pose=obj_pose, goal_x=goal_x, render=True
            )
        elif cfg.train.mode == "domain_rand":
            cfg.env.obj_pos_noise = True
            action_mean, action_std = train_cem_policy(
                cfg, zeta=None, obj_pose=None, goal_x=goal_x, render=True
            )

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
            os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "task", "policy"),
        )

    # Evaluate policy
    eval_env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed + 100,
        device_id=cfg.gpu_id,
        collision=False,
    )

    if zeta is not None:
        eval_env.set_parameters(zeta)
    if obj_pose is not None:
        eval_env.set_obj_pose(obj_pose)

    reward, imgs = collect_rollout(
        eval_env,
        cfg,    
        action,
        goal_x=goal_x,
        render=True,
    )
    imageio.mimsave(
        os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "task", "cem_final.mp4"),
        np.stack(imgs),
    )
    print(
        f"FINAL real zeta {sysid_dict['mu']} EXP {cfg.exp_id} ALGO {cfg.train.mode} reward: {reward} act {action} act_mean {action_mean} act_std {action_std}"
    )


if __name__ == "__main__":
    run_experiment()
