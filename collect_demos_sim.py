import argparse
import datetime
import os
import time

import hydra
import imageio
import joblib
import numpy as np
from tqdm import tqdm

from helpers.experiment import hydra_to_dict, set_random_seed, setup_wandb
from robot.rlds_wrapper import (convert_rlds_to_np, load_rlds_dataset,
                                wrap_env_in_rlds_logger)
from robot.robot_env import RobotEnv

from mp_env import collect_demo_pick_up

@hydra.main(config_path="configs/", config_name="collect_sim", version_base="1.1")
def run_experiment(cfg):

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--exp", type=str)
    # parser.add_argument("--logdir", type=str, default="data")
    # # hardware
    # parser.add_argument("--dof", type=int, default=6, choices=[3, 4, 6])
    # parser.add_argument(
    #     "--robot_type", type=str, default="panda", choices=["panda", "fr3"]
    # )
    # parser.add_argument(
    #     "--ip_address",
    #     type=str,
    #     default=None,
    #     choices=[None, "localhost", "172.16.0.1"],
    # )
    # parser.add_argument(
    #     "--camera_model", type=str, default="realsense", choices=["realsense", "zed"]
    # )
    # # trajectories
    # parser.add_argument("--episodes", type=int, default=10)
    # parser.add_argument("--max_episode_length", type=int, default=1000)

    # args = parser.parse_args()

    assert cfg.exp_id is not None, "Specify --exp_id"
    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "sim")
    os.makedirs(logdir, exist_ok=True)

    from robot.sim.vec_env.asid_vec import make_env, make_vec_env
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
    )

    # env = RobotEnv(
    #     control_hz=10,
    #     DoF=args.dof,
    #     robot_type=args.robot_type,
    #     ip_address=args.ip_address,
    #     camera_model=args.camera_model,
    #     max_path_length=args.max_episode_length,
    # )

    with wrap_env_in_rlds_logger(env, cfg.exp_id, logdir, max_episodes_per_shard=1) as rlds_env:
        for i in range(cfg.episodes):

            obss = []
            acts = []

            # obs = rlds_env.reset()

            # for j in tqdm(
            #     range(cfg.max_episode_length), desc=f"Collecting Trajectory {i}"
            # ):

            #         # TODO add MP / heuristic here
            #         act = rlds_env.action_space.sample()

            #         next_obs, rew, done, _ = rlds_env.step(rlds_env.type_action(act))

            #         obss.append(obs)
            #         acts.append(act)

            #         obs = next_obs
            
            success, _ = collect_demo_pick_up(rlds_env, z_waypoints=[0.3, 0.2, 0.12], noise_std=[5e-2, 1e-2, 5e-3], render=False)

            print(f"Recorded Trajectory {i}")

    env.reset()

    # check if dataset was saved
    loaded_dataset = load_rlds_dataset(logdir)

    print(f"Finished Collecting {i} Trajectories")

if __name__ == "__main__":
    run_experiment()