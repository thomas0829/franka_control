import time

import hydra
import imageio
import numpy as np
import torch

from helpers.experiment import hydra_to_dict, set_random_seed, setup_wandb
from helpers.transformations import euler_to_quat, euler_to_rmat, quat_to_euler
from robot.sim.vec_env.asid_vec import make_env, make_vec_env


def move_to_cartesian_pose(target_pose, env):
    while True:
        robot_state, _ = env.unwrapped._robot.get_robot_state()
        q_desired = env.unwrapped._robot._ik_solver.cartesian_position_to_joint_position(target_pose[:3], target_pose[3:], robot_state)
        env.unwrapped._robot.update_joints(q_desired, blocking=True)
        error = np.linalg.norm(env.unwrapped._robot.get_ee_pose()[0]- target_pose[:3])
        print(error)
        if error < 5e-2:
            break

@hydra.main(config_path="configs/", config_name="collect_sim", version_base="1.1")
def run_experiment(cfg):
    
    cfg.robot.on_screen_rendering = True
    cfg.env.obj_pos_noise = False

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
    )
    env.reset()

    imgs = []

    # MOVE ABOVE ROD
    target_pose = env.get_obj_pose().copy()
    # set fixed height
    target_pose[2] = 0.4
    # set rod z angle + z offset for franka EE
    target_pose[3:6] = quat_to_euler(env.get_obj_pose()[3:])
    # WARNING!!!! this has to be -np.pi/4 for CUBE!
    # target_pose[5] -= np.pi / 4
    # overwrite x,y angle w/ gripper default
    target_pose[3:5] = env.unwrapped._default_angle[:2]
    # set dummy gripper
    target_pose[-1] = 0

    # get rotation matrix from cube angle
    rod_angle_z = np.zeros(3)
    rod_angle_z[2] = quat_to_euler(env.get_obj_pose()[3:])[2]

    # rmat = euler_to_rmat(rod_angle_z)
    # # transform cube pos to origin
    # rod_pos_origin = target_pose[:3] @ np.linalg.inv(rmat)
    # # transform back
    # rod_pos_new = rod_pos_origin @ rmat
    # # overwrite x,y
    # target_pose[:2] = rod_pos_new[:2]
    
    target_pose[2] = 0.2
    while True:
        curr_pose = np.concatenate(env.unwrapped._robot.get_ee_pose())
        vel_act = np.zeros(7)
        lim = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        vel_act[:6] = np.clip(target_pose[:6] - curr_pose[:6], -lim, lim)
        # vel_act[3:6] = 0.
        # convert vel to delta actions
        delta_act = env.unwrapped._robot._ik_solver.cartesian_velocity_to_delta(vel_act)
        delta_gripper = env.unwrapped._robot._ik_solver.gripper_velocity_to_delta(
            vel_act[-1:]
        )
        act = np.concatenate((delta_act, delta_gripper))
        act[:6] = np.clip(target_pose[:6] - curr_pose[:6], -.1, .1)

        env.step(act)
        env.render()
        print(act[:3], np.linalg.norm(env.unwrapped._robot.get_ee_pose()[0]- target_pose[:3]))
        error = np.linalg.norm(env.unwrapped._robot.get_ee_pose()[0]- target_pose[:3])
        if error < 5e-2:
            break

    target_pose[2] = 0.08
    while True:
        curr_pose = np.concatenate(env.unwrapped._robot.get_ee_pose())
        vel_act = np.zeros(7)
        lim = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        vel_act[:6] = np.clip(target_pose[:6] - curr_pose[:6], -lim, lim)
        # vel_act[3:6] = 0.
        # convert vel to delta actions
        delta_act = env.unwrapped._robot._ik_solver.cartesian_velocity_to_delta(vel_act)
        delta_gripper = env.unwrapped._robot._ik_solver.gripper_velocity_to_delta(
            vel_act[-1:]
        )
        act = np.concatenate((delta_act, delta_gripper))
        act[:6] = np.clip(target_pose[:6] - curr_pose[:6], -.1, .1)

        env.step(act)
        env.render()
        print(act[:3], np.linalg.norm(env.unwrapped._robot.get_ee_pose()[0]- target_pose[:3]))
        error = np.linalg.norm(env.unwrapped._robot.get_ee_pose()[0]- target_pose[:3])
        if error < 5e-2:
            break

    for i in range(3):
        act = np.zeros(7)
        act[-1] = 1.
        env.step(act)
        time.sleep(0.1)
        env.render()

    # move_to_cartesian_pose(target_pose, env)
    # env.render()

    # # MOVE UP
    target_pose[2] = 0.2
    while True:
        curr_pose = np.concatenate(env.unwrapped._robot.get_ee_pose())
        vel_act = np.zeros(7)
        lim = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        vel_act[:6] = np.clip(target_pose[:6] - curr_pose[:6], -lim, lim)
        vel_act[-1] = 1.
        # vel_act[3:6] = 0.
        # convert vel to delta actions
        delta_act = env.unwrapped._robot._ik_solver.cartesian_velocity_to_delta(vel_act)
        delta_gripper = env.unwrapped._robot._ik_solver.gripper_velocity_to_delta(
            vel_act[-1:]
        )
        act = np.concatenate((delta_act, delta_gripper))
        act[:6] = np.clip(target_pose[:6] - curr_pose[:6], -.1, .1)

        env.step(act)
        env.render()
        print(act[:3], np.linalg.norm(env.unwrapped._robot.get_ee_pose()[0]- target_pose[:3]))
        error = np.linalg.norm(env.unwrapped._robot.get_ee_pose()[0]- target_pose[:3])
        if error < 5e-2:
            break

    target_pose[2] = 0.3
    while True:
        curr_pose = np.concatenate(env.unwrapped._robot.get_ee_pose())
        vel_act = np.zeros(7)
        lim = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        vel_act[:6] = np.clip(target_pose[:6] - curr_pose[:6], -lim, lim)
        vel_act[-1] = 1.
        # vel_act[3:6] = 0.
        # convert vel to delta actions
        delta_act = env.unwrapped._robot._ik_solver.cartesian_velocity_to_delta(vel_act)
        delta_gripper = env.unwrapped._robot._ik_solver.gripper_velocity_to_delta(
            vel_act[-1:]
        )
        act = np.concatenate((delta_act, delta_gripper))
        act[:6] = np.clip(target_pose[:6] - curr_pose[:6], -.1, .1)

        env.step(act)
        env.render()
        print(act[:3], np.linalg.norm(env.unwrapped._robot.get_ee_pose()[0]- target_pose[:3]))
        error = np.linalg.norm(env.unwrapped._robot.get_ee_pose()[0]- target_pose[:3])
        if error < 5e-2:
            break
    # imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)


if __name__ == "__main__":
    run_experiment()