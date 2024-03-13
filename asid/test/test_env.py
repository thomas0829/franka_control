import time

import hydra
import numpy as np
import torch

from asid.utils.move import jump_to_cartesian_pose
from asid.wrapper.asid_vec import make_env, make_vec_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb

from utils.transformations_mujoco import *

@hydra.main(config_path="../configs/", config_name="explore_puck_sim", version_base="1.1")
def run_experiment(cfg):

    # cfg.robot.DoF = 2
    cfg.robot.on_screen_rendering = True
    cfg.robot.gripper = False
    # # cfg.robot.ip_address = "172.16.0.1"
    # # cfg.robot.camera_model = "zed"

    # cfg.env.obj_id = "puck"
    # cfg.env.flatten = True

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed,
        device_id=0,
        verbose=False,
    )

    # env.set_parameters(np.zeros(1))
    env.reset()
    rod_pose = env.get_obj_pose()
    env.unwrapped.set_obj_pose(rod_pose)
    # goal = np.array([0.3, 0.])
    # while np.linalg.norm(env.unwrapped._robot.get_ee_pose()[:2] - goal) > 5e-2:
    #     env.step(goal - env.unwrapped._robot.get_ee_pose()[:2])
    #     env.render()
    # print("reached init")

    start = time.time()
    for i in range(15):
        env.seed(i)
        env.reset()
        for i in range(20):
            act = env.action_space.sample()
            act = np.zeros_like(act)
            act[0] = 0.1 if i < 10 else -0.1
            # act[0] = 0.3
            obs, reward, done, info = env.step(act)
            env.render()
            print(env.get_parameters(), reward)
    print(time.time() - start)


    target_pos = rod_pose.copy()[:3]
    target_euler = quat_to_euler_mujoco(rod_pose.copy()[3:])
    target_pose = np.concatenate((target_pos, target_euler))
    target_pose[3:] = env.unwrapped._default_angle

    target_pose[2] = 0.14
    gripper = 0.
    jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        render=False,
        verbose=True,
    )

    env.render()

    for _ in range(3):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    env.render()

    target_pose[2] = 0.4
    gripper = 1.
    jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        render=True,
        verbose=True,
    )

    env.render()

    import mujoco

    for _ in range(5):
        env.unwrapped._robot.update_gripper(1.)
        env.unwrapped._robot.data.qvel[env.unwrapped._robot.franka_joint_ids[-2]] = 10.
        mujoco.mj_step(env.unwrapped._robot.model, env.unwrapped._robot.data, nstep=env.unwrapped._robot.frame_skip)
        env.render()
    env.unwrapped._robot.update_gripper(0.)
    env.unwrapped._robot.data.qvel[env.unwrapped._robot.franka_joint_ids[-2]] = 0.
    mujoco.mj_step(env.unwrapped._robot.model, env.unwrapped._robot.data, nstep=env.unwrapped._robot.frame_skip)
    env.render()


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

