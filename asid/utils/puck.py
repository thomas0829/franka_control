import numpy as np
import mujoco


# move to strike pose
def move_to_puck(env, error_threshold=1e-2, max_steps=100, render=False):

    imgs = []
    target_pose = env.get_obj_pose()[:2]
    target_pose[0] -= 0.05 if env.unwrapped.sim else 0.05
    curr_pose = env.unwrapped._robot.get_ee_pose()[:2]
    error = np.linalg.norm(target_pose - curr_pose)

    steps = 0
    while error > error_threshold and steps < max_steps:
        steps += 1
        # print(steps, error, error > error_threshold, steps < max_steps)
        act = target_pose - curr_pose
        env.step(act)
        curr_pose = env.unwrapped._robot.get_ee_pose()[:2]
        error = np.linalg.norm(target_pose - curr_pose)
        if render:
            imgs.append(env.render())

    return imgs


def collect_rollout(env, cfg, action, goal_x, verbose=False, render=False):

    pre_reset_env_mod(env, cfg, explore=True)
    env.reset()
    post_reset_env_mod(env, cfg, mod_ik=False)
    
    imgs = []
    # move to strike pos
    tmp = move_to_puck(env)
    imgs.extend(tmp)

    post_reset_env_mod(env, cfg)

    # strike
    for i in range(1):
        env.step(np.array([action, 0.0]))
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
        env.step(np.array([-0.01, 0.0]))
        if render:
            tmp = [env.render()]
            imgs.extend(tmp)

    # compute reward
    reward = 0.0
    if goal_x is not None:
        obj_pose = env.get_obj_pose()
        reward = -np.linalg.norm(obj_pose[0] - goal_x)

    return reward, imgs


def pre_reset_env_mod(env, cfg=None, explore=False):

    # overwrite default reset
    if explore:
        env.unwrapped._reset_joint_qpos = np.array(
            # [
            #     0.1777045,
            #     0.17176202,
            #     -0.1835635,
            #     -2.78591275,
            #     0.01774895,
            #     2.82082129,
            #     0.76474708,
            # ]
             [0.16874725,  0.15265729, -0.19036844, -2.81305027, -0.014063  ,
        2.80572796,  0.77630925]
            # [
            #         0.36586183,
            #         0.35292763,
            #         -0.30332268,
            #         -2.77233706,
            #         0.09988396,
            #         2.89770401,
            #         0.74832614,
            #     ]
        )
    else:
        env.unwrapped._reset_joint_qpos = np.array(
            [
                0.1859751,
                0.44320866,
                -0.18454474,
                -2.29086876,
                0.01971104,
                2.54522133,
                0.79072034,
            ]
        )


def post_reset_env_mod(env, cfg, mod_ik=True):
    # overwrite default limits
    z = 0.14 # 0.13 if cfg.robot.ip_address is None else 0.14
    env.unwrapped.ee_space.low[2], env.unwrapped.ee_space.high[2] = z, z
    env.unwrapped.ee_space.high[0] = 1.
    env.unwrapped.action_space.low[:] = -0.2
    env.unwrapped.action_space.high[:] = 10.0

    # overwrite IK to allow for higher velocities
    from robot.real.inverse_kinematics.robot_ik_solver import RobotIKSolver

    if mod_ik:
        env.unwrapped._robot._ik_solver = RobotIKSolver(
            robot_type=cfg.robot.robot_type, control_hz=cfg.robot.control_hz, SPEED=True
        )
