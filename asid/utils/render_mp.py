import time
import numpy as np

from utils.transformations import *
from utils.transformations_mujoco import *

from asid.utils.puck import pre_reset_env_mod, post_reset_env_mod

def move_to_puck(env, viewer, error_threshold=1e-2, max_steps=100, render=False):

    imgs = []
    target_pose = env.get_obj_pose()[:2]
    target_pose[0] -= 0.05 if env.unwrapped.sim else 0.05
    curr_pose = env.unwrapped._robot.get_ee_pose()[:2]
    error = np.linalg.norm(target_pose - curr_pose)

    steps = 0
    while error > error_threshold and steps < max_steps:
        viewer.update()
        steps += 1
        # print(steps, error, error > error_threshold, steps < max_steps)
        act = target_pose - curr_pose
        env.step(act)
        curr_pose = env.unwrapped._robot.get_ee_pose()[:2]
        error = np.linalg.norm(target_pose - curr_pose)
        if render:
            imgs.append(env.render())
        viewer.render()
        a = 5
    return imgs

def collect_rollout_puck(env, viewer, cfg, action, goal_x, verbose=False, render=False):

    pre_reset_env_mod(env, cfg, explore=True)
    env.reset()
    post_reset_env_mod(env, cfg, mod_ik=False)
    
    imgs = []
    # move to strike pos
    tmp = move_to_puck(env, viewer)
    imgs.extend(tmp)

    post_reset_env_mod(env, cfg)

    # strike
    for i in range(1):
        viewer.update()
        env.step(np.array([action, 0.0]))
        viewer.render()
        # curr = env.unwrapped._robot.get_ee_pose()
        # curr[0] += action
        # curr[1:] = np.zeros(curr[1:].shape)
        # env.unwrapped._update_robot(
        #     np.concatenate((curr, [1.])),
        #     action_space="cartesian_position",
        # )
        # ids = env.unwrapped._robot.franka_joint_ids
        # qvel_des = np.zeros_like(ids)
        # qvel_des[-2] = 10.
        # env.unwrapped._robot.data.qvel[ids] = qvel_des
        # mujoco.mj_step(env.unwrapped._robot.model, env.unwrapped._robot.data, nstep=env.unwrapped._robot.frame_skip)
        if render:
            tmp = [env.render()]
            imgs.extend(tmp)

    # step sim until puck stops moving
    for i in range(20):
        viewer.update()
        env.step(np.array([-0.01, 0.0]))
        viewer.render()
        if render:
            tmp = [env.render()]
            imgs.extend(tmp)

    # compute reward
    reward = 0.0
    if goal_x is not None:
        obj_pose = env.get_obj_pose()
        reward = -np.linalg.norm(obj_pose[0] - goal_x)

    return reward, imgs

def jump_to_cartesian_pose(
    target_pose,
    gripper,
    env,
    viewer,
    render=False,
    verbose=False,
    **kwargs,
):

    target_pose = target_pose.copy()

    if target_pose[5] > np.pi / 2:
        target_pose[5] -= np.pi
    if target_pose[5] < -np.pi / 2:
        target_pose[5] += np.pi

    steps = 0

    if gripper:
        env.unwrapped._robot.update_gripper(gripper)

    imgs = []
    ctr = 0
    error = 1.
    while (
        # np.linalg.norm(target_pose[:3] - env.unwrapped._robot.get_ee_pose()[:3]) > 5e-2
        error > 2e-2
        and ctr < 300 # 100
    ):
        ctr += 1
        # while np.linalg.norm(target_pose - env.unwrapped._robot.get_ee_pose()) > 5e-2:
        robot_state = env.unwrapped._robot.get_robot_state()[0]
        desired_qpos = (
            env.unwrapped._robot._ik_solver.cartesian_position_to_joint_position(
                target_pose[:3], euler_to_quat(target_pose[3:]), robot_state
            )

        )
        # env.unwrapped._robot.move_to_joint_positions(desired_qpos)
        env.unwrapped._robot.update_desired_joint_positions(desired_qpos.tolist())
        viewer.update()
        viewer.render()
        if render:
            imgs.append(env.render())
        if verbose:
            print(
                "error", error,
            )
        # steps += 1
        
        pos_diff = target_pose[:3] - env.unwrapped._robot.get_ee_pose()[:3]
        angle_diff = target_pose[3:] - env.unwrapped._robot.get_ee_pose()[3:]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        error = np.linalg.norm(pos_diff) + np.linalg.norm(angle_diff)

    return imgs, steps


def collect_rollout_rod(env, viewer, action, control_hz=10, render=False, verbose=False, second=False,):

    env.reset()

    # if not env.unwrapped.sim:
    #     for i in range(25):
    #         rod_pose = env.get_obj_pose()
    #         time.sleep(1 / env.unwrapped._robot.control_hz)
    rod_pose = env.get_obj_pose()

    imgs = []
    reward = 0.0

    # get initial target pose
    target_pos = rod_pose.copy()[:3]
    target_euler = quat_to_euler_mujoco(rod_pose.copy()[3:])
    target_pose = np.concatenate((target_pos, target_euler))

    curr_pose = env.unwrapped._robot.get_ee_pose()
    # angular_diff = np.arctan2(np.sin(target_pose[3:] - curr_pose[3:]), np.cos(target_pose[3:] - curr_pose[3:]))
    # target_pose[3:] = curr_pose[3:] + angular_diff

    # overwrite x,y angle w/ gripper default
    target_pose[3:5] = env.unwrapped._default_angle[:2]

    # target_pose[3:5] = env.unwrapped._default_angle[:3]
    # target_pose[5] -= np.pi / 4
    # target_pose[5] = np.clip(target_pose[5], -np.pi/2, np.pi/2)

    # randomize grasp angle z
    # rng = np.random.randint(0, 3)
    # target_pose[5:] += np.pi / 2 if rng else -np.pi / 2 if rng == 1 else 0
    # target_pose[3:] = np.arctan2(np.sin(target_pose[3:]), np.cos(target_pose[3:]))

    # WARNING: real robot EE is offset by 90 deg -> target_pose[5] += np.pi / 4

    init_rod_pitch = target_euler[1]
    init_rod_yaw = target_euler[2]

    # # up right ee
    # target_orn[0] -= np.pi

    # align pose -> grasp pose
    target_pose[5] += np.pi / 2

    # real robot offset
    if not env.unwrapped.sim:
        target_pose[5] -= np.pi / 4

    # IMPORTANT: flip yaw angle mujoco to curobo!
    # target_orn[2] = -target_orn[2]

    # gripper is symmetric
    if target_pose[5] > np.pi / 2:
        target_pose[5] -= np.pi
    if target_pose[5] < -np.pi / 2:
        target_pose[5] += np.pi

    # Set grasp target to center of mass
    com = action  # 0.0381 # -0.0499
    target_pose[0] -= com * np.sin(init_rod_yaw)
    target_pose[1] += com * np.cos(init_rod_yaw)

    # MOVE ABOVE
    target_pose[2] = 0.3
    gripper = 0.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        viewer,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE DOWN
    target_pose[2] = 0.2
    gripper = 0.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        viewer,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE DOWN
    # # go down manually -> better than MP
    # while env.unwrapped._robot.get_ee_pose()[2] > 0.13 or (
    #     env.unwrapped.sim and env.unwrapped._robot.get_ee_pose()[2] > 0.12
    # ):
    #     env.step(np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0]))
    #     if render:
    #         imgs += [env.render()]

    target_pose[2] = 0.115 if env.unwrapped.sim else 0.125
    gripper = 0.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        viewer,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # GRASP
    # make sure gripper is fully closed -> 5 steps
    for _ in range(5):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        if render:
            imgs += [env.render()]
        viewer.update()
        viewer.render()

    # MOVE UP
    target_pose[2] = 0.2
    gripper = 1.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        viewer,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE UP
    target_pose[2] = 0.3
    gripper = 1.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        viewer,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # # middle of the road reward
    # if env.unwrapped.sim:
    #     rod_orn = quat_to_euler_mujoco(env.get_obj_pose()[-4:])
    #     reward += -np.linalg.norm(rod_orn[0] - init_rod_pitch)

    # MOVE TO PLACE LOCATION
    target_pose[:3] = np.array([0.4, -0.3, 0.5])
    target_pose[3:6] = env.unwrapped._default_angle
    
    if second:
        target_pose[5] += 0
    else:
        target_pose[5] -= np.pi / 2
    
    # if not env.unwrapped.sim:
    #     target_pose[5] -= np.pi / 4
    # real robot offset
    # TODO check this!
    # if not env.unwrapped.sim:
    #     target_pose[5] -= np.pi / 4
    gripper = 1.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        viewer,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE DOWN
    target_pose[2] = 0.35
    gripper = 1.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        viewer,
        render=render,
        verbose=verbose,
    )
    imgs += tmp
    
    # MOVE DOWN
    # target_pose[2] = 0.17
    target_pose[2] = 0.31 if second else 0.27
    gripper = 1.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        viewer,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # RELEASE GRASP
    for _ in range(1):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        if render:
            imgs += [env.render()]
        viewer.update()
        viewer.render()

    # MOVE UP
    target_pose[2] = 0.5
    gripper = 0.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        viewer,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # end of the road reward
    if env.unwrapped.sim:
        rod_orn = quat_to_euler_mujoco(env.get_obj_pose()[-4:])
        reward += -np.linalg.norm(rod_orn[0] - init_rod_pitch)

        # rod dropped
        if env.get_obj_pose()[2] < 0.04:
            # print("dropped", env.get_obj_pose())
            reward += -10

    return reward, imgs
