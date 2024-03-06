import numpy as np

from utils.transformations import *
from utils.transformations_mujoco import *


def jump_to_cartesian_pose(
    target_pose,
    gripper,
    env,
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

    while (
        np.linalg.norm(target_pose[:3] - env.unwrapped._robot.get_ee_pose()[:3]) > 5e-2
    ):
        # while np.linalg.norm(target_pose - env.unwrapped._robot.get_ee_pose()) > 5e-2:
        robot_state = env.unwrapped._robot.get_robot_state()[0]
        desired_qpos = (
            env.unwrapped._robot._ik_solver.cartesian_position_to_joint_position(
                target_pose[:3], euler_to_quat(target_pose[3:]), robot_state
            )
        )
        # env.unwrapped._robot.move_to_joint_positions(desired_qpos)
        env.unwrapped._robot.update_desired_joint_positions(desired_qpos.tolist())
        if render:
            env.render()
        if verbose:
            print(
                "error",
                "pos",
                np.linalg.norm(
                    target_pose[:3] - env.unwrapped._robot.get_ee_pose()[:3]
                ),
                "all",
                np.linalg.norm(target_pose - env.unwrapped._robot.get_ee_pose()),
            )
        # steps += 1
    imgs = []
    if render:
        imgs.append(env.render())
    return imgs, steps


def collect_rollout(env, action, control_hz=10, render=False, verbose=False):

    env.reset()

    rod_pose = env.get_obj_pose()
    
    imgs = []

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
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE DOWN
    # go down manually -> better than MP
    while env.unwrapped._robot.get_ee_pose()[2] > 0.13 or (
        env.unwrapped.sim and env.unwrapped._robot.get_ee_pose()[2] > 0.12
    ):
        env.step(np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0]))
        if render:
            env.render()
        
    # GRASP
    # make sure gripper is fully closed -> 3 steps
    for _ in range(3):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        if render:
            env.render()

    # MOVE UP
    target_pose[2] = 0.2
    gripper = 1.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
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
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE TO PLACE LOCATION
    target_pose[:3] = np.array([0.4, -0.3, 0.3])
    target_pose[3:5] = env.unwrapped._default_angle[:2]
    target_pose[5] = -np.pi / 2
    # real robot offset
    # TODO check this!
    if not env.unwrapped.sim:
        target_pose[5] += np.pi / 4
    gripper = 1.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE DOWN
    target_pose[2] = 0.15
    gripper = 1.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # RELEASE GRASP
    for _ in range(1):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # MOVE UP
    target_pose[2] = 0.3
    gripper = 0.0
    tmp, _ = jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    rod_orn = quat_to_euler_mujoco(env.get_obj_pose()[-4:])
    reward = -np.linalg.norm(rod_orn[0] - init_rod_pitch)
    return reward, imgs
