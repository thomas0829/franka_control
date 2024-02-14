from functools import partial

from robot.robot_env import RobotEnv
from robot.sim.mujoco.asid_wrapper import ASIDWrapper


def make_env(
    robot_cfg_dict, env_cfg_dict, seed=0, device_id=0, exp_reward=False, verbose=False
):

    if verbose:
        print("robot config", robot_cfg_dict)
        print("env config", env_cfg_dict)

    robot_cfg_dict["model_name"] = robot_cfg_dict["model_name"].replace(
        "base", env_cfg_dict["obj_id"]
    )

    env = RobotEnv(**robot_cfg_dict, device_id=device_id, verbose=verbose)
    env = ASIDWrapper(env, **env_cfg_dict)

    if exp_reward:
        env.create_exp_reward(robot_cfg_dict, env_cfg_dict, seed)
    env.seed(seed)

    return env


def make_vec_env(
    robot_cfg_dict, env_cfg_dict, num_workers, seed, device_id=0, exp_reward=False
):
    from robot.sim.vec_env.vec_wrapper import SubVecEnv

    env_fns = [
        partial(
            make_env,
            robot_cfg_dict,
            env_cfg_dict,
            seed=seed + i,
            device_id=device_id,
            verbose=bool(i == 0),
            exp_reward=exp_reward,
        )
        for i in range(num_workers)
    ]
    return SubVecEnv(env_fns)
