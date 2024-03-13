import numpy as np

def pre_reset_env_mod(env, cfg):
    # overwrite default reset
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

def post_reset_env_mod(env, cfg):
    # overwrite default limits
    env.unwrapped.ee_space.low[2], env.unwrapped.ee_space.high[2] = 0.13, 0.13
    env.unwrapped.ee_space.high[0] = 0.6
    env.unwrapped.action_space.low[:] = -0.2
    env.unwrapped.action_space.high[:] = 10.

    # overwrite IK to allow for higher velocities
    from robot.real.inverse_kinematics.robot_ik_solver import RobotIKSolver
    env.unwrapped._robot._ik_solver = RobotIKSolver(robot_type=cfg.robot.robot_type, control_hz=cfg.robot.control_hz, SPEED=True)