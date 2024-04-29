import json
import os

import hydra
import matplotlib.pyplot as plt
from nvisii_renderer import NVISIIRenderer

from asid.utils.render_mp import *
from asid.wrapper.asid_vec import make_env
from robot.robot_env import RobotEnv
from utils.experiment import hydra_to_dict
from utils.transformations import *
from utils.transformations_mujoco import *


@hydra.main(
    config_path="../asid/configs/", config_name="explore_rod_sim", version_base="1.1"
)
def run_experiment(cfg):

    cfg.asid.reward = False

    robot_cfg_dict = hydra_to_dict(cfg.robot)
    robot_cfg_dict["model_name"] = "rod_franka"
    
    if "rod" in robot_cfg_dict["model_name"]:

        # cfg.robot.DoF = 6
        # cfg.robot.gripper = True 
        # cfg.env.obj_pose_noise_dict = None
        env = make_env(
            robot_cfg_dict=hydra_to_dict(cfg.robot),
            env_cfg_dict=hydra_to_dict(cfg.env),
            asid_cfg_dict=hydra_to_dict(cfg.asid),
            seed=cfg.seed,
            device_id=0,
            verbose=True,
        )
        env.set_parameters(np.ones(1) * 0.1)
        # # explore
        env.set_obj_pose(np.array([0.4, 0.3, 0.02, 0.93937271, 0.0, 0.0, -0.34289781]))
     
    # elif "puck" in robot_cfg_dict["model_name"]:
    #     cfg.env.obj_pose_noise_dict = None
    #     cfg.robot.DoF = 2
    #     cfg.robot.gripper = False 
    #     cfg.robot.max_path_length = 1e5
    #     env = make_env(
    #         robot_cfg_dict=hydra_to_dict(cfg.robot),
    #         env_cfg_dict=hydra_to_dict(cfg.env),
    #         asid_cfg_dict=hydra_to_dict(cfg.asid),
    #         seed=cfg.seed,
    #         device_id=0,
    #         verbose=True,
    #     )

    # else:
    # env = RobotEnv(**robot_cfg_dict, device_id=0, verbose=True)

    if "sphere" in robot_cfg_dict["model_name"]:
        env._reset_joint_qpos = np.array(
            [
                0.50563447,
                0.60037293,
                -0.20052734,
                -2.18004633,
                0.29837491,
                2.76770274,
                1.67566914,
            ]
        )
    elif "articulation" in robot_cfg_dict["model_name"]:
        env._reset_joint_qpos = np.array(
            [
                0.00337199,
                -0.02690877,
                -0.00988887,
                -2.33686253,
                0.01294934,
                2.35197591,
                0.11204342,
            ]
        )
    elif "puck" in robot_cfg_dict["model_name"]:
        env._reset_joint_qpos = np.array(
            [
                0.16874725,
                0.15265729,
                -0.19036844,
                -2.81305027,
                -0.014063,
                2.80572796,
                0.77630925,
            ]
        )
    elif "rod" in robot_cfg_dict["model_name"]:
        env._reset_joint_qpos = np.array(
            [
                0.85290707,
                0.29776727,
                0.0438237,
                -2.70994978,
                -0.00481878,
                2.89241547,
                1.67766532,
            ]
        )

    env.reset()

    with open(os.path.join("rendering", "nvisii_config.json")) as f:
        renderer_config = json.load(f)

    renderer_config["img_path"] = os.path.join(
        renderer_config["img_path"], robot_cfg_dict["model_name"]
    )
    viewer = NVISIIRenderer(env=env.unwrapped._robot, **renderer_config)
    viewer.reset()

    pos = [1.1, 0.5, 0.8]
    at = [0, -0.2, 0]
    
    # if "puck" in robot_cfg_dict["model_name"]:
    #     pos = [0.9, 0.8, 0.8]
    #     at = [0.6, 0.0, 0.1]
    # elif "rod" in robot_cfg_dict["model_name"]:
    #     pos = [0.9, 0.5, 0.6]
    #     at = [0, -0.4, 0.2]
    # elif "sphere" in robot_cfg_dict["model_name"]:
    #     pos = [0.9, 0.7, 0.7]
    #     at = [0.4, -0.1, 0.07]

    viewer.set_camera_pos_quat(pos, at)

    # if "puck" in robot_cfg_dict["model_name"]:
    #     action = 0.14
    #     collect_rollout_puck(env, viewer, cfg, action, 0.5, render=True)
    # elif "rod" in robot_cfg_dict["model_name"]:
    #     action = 0.06
    #     collect_rollout_rod(env, viewer, action, render=True)

    for i in range(10):
        viewer.update()

        if "rod" in robot_cfg_dict["model_name"]:
            env.step(np.array([0.1, 0.05]))
            # env.reset()
            # env.step(env.action_space.sample())

        elif "puck" in robot_cfg_dict["model_name"]:
            env.step(np.array([0.1 + env.action_space.sample()[0] / 3 if i < 4 else 0., 0.0]))
        else:
            env.step(env.action_space.sample())
        # plt.imshow(env.render())
        viewer.render()

    viewer.close()


if __name__ == "__main__":
    run_experiment()
