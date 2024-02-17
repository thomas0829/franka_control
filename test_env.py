import time

import hydra
import numpy as np
import torch

from helpers.experiment import hydra_to_dict, set_random_seed, setup_wandb
from robot.sim.vec_env.asid_vec import make_env, make_vec_env


@hydra.main(config_path="configs/", config_name="collect_sim", version_base="1.1")
def run_experiment(cfg):
    
    cfg.robot.DoF = 6
    cfg.robot.on_screen_rendering = False
    cfg.robot.gripper = False
    # cfg.robot.ip_address = "172.16.0.1"
    # cfg.robot.camera_model = "zed"

    cfg.env.obj_pos_noise = False

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
        exp_reward=False,
        verbose=True,
    )
    env.reset()

    # env.set_parameters(np.array([0., 1.]))
    # env.unwrapped.get_images_and_points()
    # import joblib
    # joblib.dump(env.unwrapped.get_images_and_points(), "points_sim")

    while True:
        obs, reward, done, info = env.step(env.action_space.sample())
        # obs, reward, done, info = env.step(np.array([0.3, 0., 1.]))
        # obs, reward, done, info = env.step(np.array([0., 0.]))
        # time.sleep(0.3)
        # print(env.unwrapped._robot.get_ee_pos(), env.get_obj_pose()[:3], reward)
        env.render()
        

if __name__ == "__main__":
    run_experiment()