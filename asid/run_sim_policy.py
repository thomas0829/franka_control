import argparse
import datetime
import os
import time

import imageio
import joblib
import numpy as np
import torch
from torchcontrol.policies import CartesianImpedanceControl

from perception.trackers.color_tracker import ColorTracker
from robot.robot_env import RobotEnv
from utils.pointclouds import crop_points
from utils.transformations import euler_to_quat, quat_to_euler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--exp", type=str)
    parser.add_argument("--save_dir", type=str, default="data")

    # hardware
    parser.add_argument("--dof", type=int, default=3, choices=[2, 3, 4, 6])
    parser.add_argument(
        "--robot_type", type=str, default="panda", choices=["panda", "fr3"]
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        default=None,
        choices=[None, "localhost", "172.16.0.1"],
    )
    parser.add_argument(
        "--camera_model", type=str, default="realsense", choices=["realsense", "zed"]
    )

    # training
    parser.add_argument("--max_episode_length", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    args.exp = "test"
    assert args.exp is not None, "Specify --exp"
    device = torch.device(
        ("cuda:" + str(args.gpu_id))
        if args.gpu_id >= 0.0 and torch.cuda.is_available()
        else "cpu"
    )

    control_hz = 10

    cfg = {
        "control_hz": control_hz,
        "DoF": args.dof,
        "robot_type": args.robot_type,
        "gripper": False,
        "ip_address": args.ip_address,
        "camera_model": args.camera_model,
        "camera_resolution": (480, 480),
        "max_path_length": args.max_episode_length,
        "on_screen_rendering": False,
    }

    env = RobotEnv(**cfg)

    tracker = ColorTracker(outlier_removal=True)
    tracker.reset()
    # define workspace
    crop_min = [0.0, -0.4, -0.1]
    crop_max = [0.5, 0.4, 0.5]

    # custom reset pose
    obs = env.reset()

    # env._robot.update_command(_reset_joint_qpos, action_space="joint_position", blocking=True)
    # obs = env.get_observation()

    from gym.spaces import Box

    env.observation_space = Box(-np.ones(16), np.ones(16))

    # TODO REMOVE
    ee_quat = euler_to_quat(env._robot.get_ee_angle().copy())

    poses = [
        np.array([0.35, -0.1, 0.2]),
        np.array([0.5, -0.3, 0.3]),
        np.array([0.5, -0.3, 0.3]),
    ]

    env._robot.has_offscreen_renderer = True
    env._robot.has_renderer = False

    pose = []
    imgs = []
    # for i in range(args.max_episode_length):
    for i in range(500):
        # while True:
        # PREDICT
        # obs_tmp = np.concatenate((obs["lowdim_ee"][:2], rod_pose, obs["lowdim_qpos"][:-1]))
        # actions, _state = model.predict(obs_tmp, deterministic=False)

        # ACT
        actions = np.array([0.0, 1.0, 0.0])
        # actions = np.random.uniform([-1., -1., 0.], [1., 1., 0.], size=3)

        # # ee_pos = np.array([0.5, 0.3, 0.3])
        # ee_pos = env._robot.get_ee_pos().copy() + actions # + np.random.uniform(-1e-3, 1e-3, size=3)
        # qpos = env._robot._ik_solver.cartesian_position_to_joint_position(ee_pos, ee_quat, env._robot.get_robot_state()[0])
        # env._robot.update_command(
        #     np.concatenate((qpos, np.zeros((1,)))), action_space="joint_position", blocking=False
        # )
        # qpos = env._robot._ik_solver.cartesian_velocity_to_joint_velocity(np.concatenate((actions,np.zeros(4))), env._robot.get_robot_state()[0])
        # env._robot.update_command(
        #     np.concatenate((qpos, np.zeros((1,)))), action_space="cartesian_velocity", blocking=False
        # )

        # Cartesian Impedance PD
        # env._robot.policy = CartesianImpedanceControl(
        #     joint_pos_current=torch.tensor(
        #         env._robot.get_joint_positions(), dtype=torch.float32
        #     ),
        #     Kp=1.
        #     * torch.tensor(env._robot.metadata["default_Kx"], dtype=torch.float64),
        #     Kd=1.
        #     * torch.tensor(env._robot.metadata["default_Kxd"], dtype=torch.float64),
        #     robot_model=env._robot.toco_robot_model,
        #     ignore_gravity=True,
        # )
        # # zero out d, low q, increase q until stable, gradually increase d
        # env._robot.policy.ee_pos_desired = torch.nn.Parameter(
        #     torch.tensor(env._robot.get_ee_pos().copy() + actions)
        # )
        # env._robot.policy.ee_quat_desired = torch.nn.Parameter(torch.tensor(ee_quat))
        # env._robot.policy.ee_vel_desired = torch.nn.Parameter(
        #     torch.tensor(torch.zeros(3))
        # )
        # env._robot.policy.ee_rvel_desired = torch.nn.Parameter(
        #     torch.tensor(torch.zeros(3))
        # )

        # output_pkt = env._robot.policy.forward(env._robot.get_robot_state()[0])
        # torques = output_pkt["joint_torques"].detach().numpy()
        # env._robot.apply_joint_torques(torques)

        # pos = ee_pos # poses[int(i // 50)]
        # ee_quat = euler_to_quat(env._robot.get_ee_angle().copy()) + np.random.uniform(-1e-3, 1e-3, size=4)
        # qpos = env._robot._ik_solver.cartesian_position_to_joint_position(pos, ee_quat, env._robot.get_robot_state()[0])
        # # qpos = env._robot.toco_robot_model.inverse_kinematics(torch.tensor(pos), torch.tensor(ee_quat))
        # env._robot.update_command(
        #     np.concatenate((qpos, np.zeros((1,)))), action_space="joint_position", blocking=False
        # )

        # udpate_pkt = {}
        # udpate_pkt["ctrl_mode"] = 0.0

        # udpate_pkt["q_desired"] = (
        #         torch.tensor(qpos)
        #     )

        # for i in range(50):
        #     output_pkt = env._robot._update_current_controller(udpate_pkt)

        #     torques = output_pkt["joint_torques"].detach().numpy()
        #     env._robot.apply_joint_torques(torques)

        # env._robot.update_desired_joint_positions(np.concatenate((qpos, np.zeros((1,)))), ctrl_mode=0.0)
        # time.sleep(0.1)
        # env._robot.render()

        pose.append(env._robot.get_ee_pose())
        # print("pose", pos, "ee", env._robot.get_ee_pos(), "diff", np.linalg.norm(env._robot.get_joint_positions() - qpos))
        # print("waypoint", qpos, "qpos", env._robot.get_joint_positions(), "diff", np.linalg.norm(env._robot.get_joint_positions() - qpos))
        # print("ee_pos", ee_pos, "ee_real", env._robot.get_ee_pos(), "diff", env._robot.get_ee_pos() - ee_pos)
        # print("ee_real", env._robot.get_ee_pos())
        next_obs, rew, done, _ = env.step(actions)
        # time.sleep(0.1)
        imgs.append(env.render())
        # obs = next_obs

        # print("ee", obs["lowdim_ee"][:2])

    env.reset()

    # np.save("real" if args.ip_address is not None else 'sim_weight', np.array(pose))
    imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)
    import matplotlib.pyplot as plt

    plt.close()
    data = np.array(pose)[:, 0]
    labels = ["x", "y", "z"]
    for i in range(3):
        plt.plot(data[:, i : i + 1], label=labels[i])
    plt.legend()
    # plt.show()
    plt.savefig("plot.png")
