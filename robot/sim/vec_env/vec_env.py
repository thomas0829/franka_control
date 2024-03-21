from functools import partial

from robot.robot_env import RobotEnv


def make_env(
    robot_cfg_dict,
    env_cfg_dict,
    seed=0,
    device_id=0,
    verbose=False,
):

    if verbose:
        print("robot config", robot_cfg_dict)

    if env_cfg_dict is not None:
        robot_cfg_dict["model_name"] = robot_cfg_dict["model_name"].replace(
            "base", env_cfg_dict["obj_id"]
        )

    env = RobotEnv(**robot_cfg_dict, device_id=device_id, verbose=verbose)

    if env_cfg_dict is not None:
        if robot_cfg_dict["ip_address"] is None:
            from robot.sim.mujoco.obj_wrapper import ObjWrapper

            env = ObjWrapper(env, **env_cfg_dict, verbose=verbose)
        else:
            from robot.real.obj_tracker_wrapper import ObjectTrackerWrapper

            env = ObjectTrackerWrapper(env, **env_cfg_dict, verbose=verbose)

    env.seed(seed)

    return env


def make_vec_env(
    robot_cfg_dict,
    env_cfg_dict,
    num_workers,
    seed,
    device_id=0,
    verbose=False,
):
    from robot.sim.vec_env.vec_wrapper import SubVecEnv

    env_fns = [
        partial(
            make_env,
            robot_cfg_dict,
            env_cfg_dict,
            seed=seed + i,
            device_id=device_id,
            verbose=bool(i == 0) and verbose,
        )
        for i in range(num_workers)
    ]
    return SubVecEnv(env_fns)
