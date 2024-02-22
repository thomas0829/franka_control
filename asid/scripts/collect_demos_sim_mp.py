import argparse
import datetime
import os
import time

import hydra
import imageio
import joblib
import numpy as np
from tqdm import tqdm

from asid.mp_env import collect_demo_pick_up
from robot.rlds_wrapper import (
    convert_rlds_to_np,
    load_rlds_dataset,
    wrap_env_in_rlds_logger,
)
from robot.robot_env import RobotEnv
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb


@hydra.main(
    config_path="../configs/", config_name="collect_cube_sim", version_base="1.1"
)
def run_experiment(cfg):

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "sim")
    os.makedirs(logdir, exist_ok=True)

    from asid.wrapper.asid_vec import make_env, make_vec_env
    
    cfg.robot.DoF = 6
    cfg.robot.gripper = True
    cfg.robot.on_screen_rendering = True
    cfg.robot.max_path_length = 100

    cfg.env.flatten = False
    cfg.env.obj_pos_noise = True

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
    )

    # for i in range(10):
    #     time.sleep(2.)
    #     success, imgs = collect_demo_pick_up(
    #                 env,
    #                 z_waypoints=[0.3, 0.2, 0.12],
    #                 noise_std=[5e-2, 1e-2, 5e-3],
    #                 render=True,
    #                 verbose=True,
    #             )
    #     time.sleep(2.)

    # imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)
    
    with wrap_env_in_rlds_logger(
        env, cfg.exp_id, logdir, max_episodes_per_shard=1
    ) as rlds_env:
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

            success, _ = collect_demo_pick_up(
                rlds_env,
                z_waypoints=[0.3, 0.2, 0.12],
                noise_std=[5e-2, 1e-2, 5e-3],
                render=True,
            )

            print(f"Recorded Trajectory {i}, success {success}")

    env.reset()

    # check if dataset was saved
    loaded_dataset = load_rlds_dataset(logdir)

    print(f"Finished Collecting {i} Trajectories")


if __name__ == "__main__":
    run_experiment()
