import time

import hydra
import numpy as np
import torch

from asid.wrapper.asid_vec import make_env, make_vec_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb


@hydra.main(config_path="../configs/", config_name="explore_rod_sim", version_base="1.1")
def run_experiment(cfg):

    cfg.robot.DoF = 2
    cfg.robot.on_screen_rendering = True
    cfg.robot.gripper = True
    # # cfg.robot.ip_address = "172.16.0.1"
    # # cfg.robot.camera_model = "zed"

    # # cfg.env.obj_pos_noise = False
    # cfg.env.flatten = True

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed,
        device_id=0,
        verbose=False,
    )

    # env.set_parameters(np.array([0., 1.]))
    # env.unwrapped.get_images_and_points()
    # import joblib
    # joblib.dump(env.unwrapped.get_images_and_points(), "points_sim")

    start = time.time()
    for i in range(15):
        env.seed(i)
        env.reset()
        for i in range(10):
            act = env.action_space.sample()
            act = np.zeros_like(act)
            # act[0] = 0.3
            obs, reward, done, info = env.step(act)
            # env.render()
            print(env.get_parameters(), reward)
    print(time.time() - start)
            # time.sleep(0.1)
            # env.render()
        # print(reward, env.get_parameters(), env.unwrapped._robot.get_ee_pos())
        # obs, reward, done, info = env.step(np.array([0.3, 0., 1.]))
        # obs, reward, done, info = env.step(np.array([0., 0.]))
        # time.sleep(0.3)
        # print(env.unwrapped._robot.get_ee_pos(), env.get_obj_pose()[:3], reward)
        # env.render()
    


if __name__ == "__main__":
    run_experiment()

