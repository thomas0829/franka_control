import glob
import os
import time

import h5py
import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from robot.robot_env import RobotEnv
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict


from utils.transformations import *
from utils.transformations_mujoco import *

@hydra.main(
    config_path="../../configs/", config_name="collect_cube_real", version_base="1.1"
)
def run_experiment(cfg):

    logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    os.makedirs(logdir, exist_ok=True)

    cfg.robot.max_path_length = cfg.max_episode_length

    cfg.robot.DoF = 6
    cfg.robot.control_hz = 1
    cfg.robot.gripper = True
    cfg.robot.blocking_control = True
    cfg.robot.on_screen_rendering = False

    # cfg.env.flatten = False

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    env.action_space.low[:-1] = -1.0
    env.action_space.high[:-1] = 1.0

    env._reset_joint_qpos = np.array([0.0, -0.8, 0.0, -2.3, 0.1, 2.3, 0.8])
    offset = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]) - np.array([0.0, -0.8, 0.0, -2.3, 0.1, 2.3, 0.8])

    dataset_path = "data/1000_demos_pick_red_test_10/tmp"
    # dataset_path = f"data/{cfg.exp_id}/train"
    file_names = glob.glob(f"{dataset_path}/*.hdf5")
    assert len(file_names) > 0, f"WARNING: no data in {dataset_path}!"

    num_trajectories = 5

    file_names = file_names[1:]
    for i, file in enumerate(file_names[:num_trajectories]):

        # episode = h5py.File(file, "r+")["data"]["demo_0"]

        # data.append(
        #     {
        #         'actions': demo['actions'],
        #         'obs': demo['obs'],
        #         "images": demo['obs']['world_camera_low_res_image']
        #     }
        # )

        # cfg.robot.blocking_control = False

        obs = env.reset()

            ### EE POSE ### -> looks like it works! verufy ...
        env._reset_joint_qpos = np.array([0.0, -0.8, 0.0, -2.3, 0.1, 2.3, 0.8])
        env.reset()
        file_idx=i
        file = file_names[file_idx]
        episode = h5py.File(file, "r+")["data"]["demo_0"]


        # OFFSET FROM WORLD FRAME TO ROBOT FRAME
        world_offset_pos = np.array([0.2045, 0., 0.])
        ee_offset_euler = np.array([0., 0., -np.pi / 4])

        # BLOCKING CONTROL BY RUNNING WITH LOWER CONTROL FREQUENCY Hz
        env.control_hz = 1

        first_grasp_idx = np.where(episode["actions"][..., -1] == -1)[0][0]
        actions = episode["actions"][:].copy()
        actions[first_grasp_idx:] = 1

        jointss = []
        poss = []
        poss_des = []
        eulers = []
        eulers_des = []
        imgs = []


        for i in range(len(episode["obs"]["eef_pos"])):

            start_time = time.time()
            
            desired_ee_pos = episode["obs"]["eef_pos"][i] + world_offset_pos
            desired_ee_quat = episode["obs"]["eef_quat"][i]
            desired_ee_euler = quat_to_euler(desired_ee_quat)
            desired_ee_euler = add_angles(ee_offset_euler, desired_ee_euler)

            gripper = actions[i,-1]

            env._update_robot(
                        np.concatenate((desired_ee_pos, desired_ee_euler, [gripper])),
                        action_space="cartesian_position",
                        blocking=False,
                    )

            jointss.append(env._robot.get_joint_positions())

            poss.append(env._robot.get_ee_pos())
            poss_des.append(desired_ee_pos)
            eulers.append(env._robot.get_ee_angle())
            eulers_des.append(desired_ee_euler)

            imgs.append(env.render())

            # SLEEP TO MAINTAIN CONTROL FREQUENCY
            comp_time = time.time() - start_time
            sleep_left = max(0, (1 / env.control_hz) - comp_time)
            time.sleep(sleep_left)

        import imageio
        # imageio.mimsave(f"ee_pose_{file_idx}.gif", np.stack(imgs), duration=5.)
        imageio.mimsave(f"ee_pose_{file_idx}.gif", np.concatenate((episode["obs"]["world_camera_low_res_image"], np.stack(imgs)), axis=2), duration=5.)
        

        plt.close()
        poss = np.stack(poss)
        poss_des = np.stack(poss_des)
        plt.plot(poss[...,0], color="tab:orange", label="x real")
        plt.plot(episode["obs"]["eef_pos"][...,0], color="tab:orange", linestyle="dotted", label="x isaac")
        #plt.plot(poss_des[...,0], color="tab:orange", linestyle="--", label="x real (commanded)")
        plt.plot(poss[...,1], color="tab:blue", label="y real")
        plt.plot(episode["obs"]["eef_pos"][...,1], color="tab:blue", linestyle="dotted", label="y isaac")
        #plt.plot(poss_des[...,1], color="tab:blue", linestyle="--", label="y real (commanded)")
        plt.plot(poss[...,2], color="tab:pink", label="z real")
        plt.plot(episode["obs"]["eef_pos"][...,2], color="tab:pink", linestyle="dotted", label="z isaac")
        #plt.plot(poss_des[...,2], color="tab:pink", linestyle="--", label="z real (commanded)")
        plt.legend()
        plt.savefig(f"ee_pos_{file_idx}.png")

        plt.close()
        eulers = np.stack(eulers)
        eulers_des = np.stack(eulers_des)
        plt.plot(eulers[...,0], color="tab:orange", label="roll real")
        plt.plot(quat_to_euler(episode["obs"]["eef_quat"])[...,0], color="tab:orange", linestyle="dotted", label="roll isaac")
        #plt.plot(eulers_des[...,0], color="tab:orange", linestyle="--", label="roll real (commanded)")
        plt.plot(eulers[...,1], color="tab:blue", label="pitch real")
        plt.plot(quat_to_euler(episode["obs"]["eef_quat"])[...,1], color="tab:blue", linestyle="dotted", label="pitch isaac")
        #plt.plot(eulers_des[...,1], color="tab:blue", linestyle="--", label="pitch real (commanded)")
        plt.plot(eulers[...,2], color="tab:pink", label="yaw real")
        plt.plot(quat_to_euler(episode["obs"]["eef_quat"])[...,2], color="tab:pink", linestyle="dotted", label="yaw isaac")
        #plt.plot(eulers_des[...,2], color="tab:pink", linestyle="--", label="yaw real (commanded)")
        plt.legend()
        plt.savefig(f"ee_angles_{file_idx}.png")

        plt.close()
        jointss = np.stack(jointss)
        colors = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
        for j in range(7):
            if j == 0:
                plt.plot(jointss[...,j], label=f"joint {j} real", color=colors[j])
                plt.plot(episode["obs"]["joint_pos"][..., j], label=f"joint {j} isaac", color=colors[j], linestyle="dashed")
            else:
                plt.plot(jointss[...,j], color=colors[j])
                plt.plot(episode["obs"]["joint_pos"][..., j], color=colors[j], linestyle="dashed")
        plt.legend()
        plt.savefig(f"joint_pos_{file_idx}.png")

        
        # desired_ee_pos = episode["obs"]["eef_pos"][0]
        # desired_ee_pos[2] -= 0.1034
        # desired_ee_quat = episode["obs"]["eef_quat"][0]
        # robot_state = env._robot.get_robot_state()[0]

        # desired_qpos = env._robot._ik_solver.cartesian_position_to_joint_position(desired_ee_pos, desired_ee_quat, robot_state)

        # env._robot.update_joints(
        #         desired_qpos.tolist(), velocity=False, blocking=True
        #     )
        
        # desired_ee_pos = episode["obs"]["eef_pos"][0]
        # desired_ee_quat = episode["obs"]["eef_quat"][0]
        # env._robot.control_hz = 10
        # env._robot.blocking_control = False
        # obs = env.reset()
        # error = np.linalg.norm(desired_ee_pos - obs["lowdim_ee"][:3])
        # while error > 1e-3:
        #     act = np.zeros(7)
        #     act[:3] = desired_ee_pos - obs["lowdim_ee"][:3]
        #     obs, _, _, _ = env.step(act)
        #     error = np.linalg.norm(desired_ee_pos - obs["lowdim_ee"][:3])
        #     print(error, desired_ee_pos, obs["lowdim_ee"][:3])



        # des_pose = env.get_observation()["lowdim_ee"].copy()
        # # des_pose[5] -= np.pi / 4
        # error = des_pose - env.get_observation()["lowdim_ee"]

        # while np.sum(error) > 1e-3:
        #     env.step(error)
        #     error = des_pose - env.get_observation()["lowdim_ee"]

        # cfg.robot.blocking_control = True

        # act = np.zeros(7)
        # act[5] = -np.pi / 4
        # env.step(act)

        # obss = []
        # acts = []

        # actions = episode["actions"]

        # # unnormalize actions
        # actions[:, :6] *= np.array(
        #     [[0.05, 0.05, 0.05, 0.17453293, 0.17453293, 0.17453293]]
        # )

        # for act in tqdm(actions):

        #     print(act)
        #     next_obs, rew, done, _ = env.step(act)

        #     obss.append(obs)
        #     acts.append(act)
        #     obs = next_obs

        # env.reset()

        # # visualize traj
        # img_obs = np.stack([obs["215122255213_rgb"] for obs in obss])
        # imageio.mimsave(os.path.join(logdir, "replay.mp4"), img_obs)
        # # ugly hack to get the demo images

        # img_demo = episode["obs"]["world_camera_low_res_image"]
        # imageio.mimsave(os.path.join(logdir, "demo.mp4"), img_demo)

        # # plot difference between demo and replay
        # plt.close()
        # ee_obs = np.stack([obs["lowdim_ee"] for obs in obss])
        # ee_act = episode["obs"]["eef_pos"]

        # labels = ["x", "y", "z"]
        # colors = ["tab:orange", "tab:blue", "tab:green"]

        # for i, (l, c) in enumerate(zip(labels, colors)):
        #     plt.plot(ee_act[:, i], color=c, label=f"{l} demo")
        #     plt.plot(ee_obs[:, i], color=c, linestyle="dashed", label=f"{l} replay")

        # plt.legend()
        # plt.show()

        # time.sleep(5)

    # env.reset()


if __name__ == "__main__":
    run_experiment()
