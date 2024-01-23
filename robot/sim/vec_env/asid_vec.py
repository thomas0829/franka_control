from functools import partial

from robot.robot_env import RobotEnv

from robot.sim.mujoco.asid_wrapper import ASIDWrapper
from robot.sim.vec_env.vec_wrapper import SubVecEnv


def make_env(cfg, seed=0, device_id=0, exp_reward=False, verbose=False):
    if verbose:
        print(cfg)

    env = RobotEnv(**cfg)
    env = ASIDWrapper(env)

    if exp_reward:
        env.create_exp_reward(cfg, seed)
    env.seed(seed)

    return env


def make_vec_env(
    cfg_dict, num_workers, seed, device_id=0, exp_reward=False, gymnasium=False
):
    env_fns = [
        partial(
            make_env,
            cfg_dict,
            seed=seed + i,
            device_id=device_id,
            verbose=bool(i == 0),
            exp_reward=exp_reward,
        )
        for i in range(num_workers)
    ]
    return SubVecEnv(env_fns, gymnasium=gymnasium)
