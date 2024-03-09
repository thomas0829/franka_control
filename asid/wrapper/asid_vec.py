from functools import partial

from robot.robot_env import RobotEnv


def make_env(
    robot_cfg_dict,
    env_cfg_dict=None,
    asid_cfg_dict=None,
    seed=0,
    device_id=0,
    sysid=False,
    collision=False,
    verbose=False,
    second=False,
):

    if verbose:
        print("robot config", robot_cfg_dict)
        if env_cfg_dict is not None:
            print("env config", env_cfg_dict)
            if asid_cfg_dict is not None:
                print("asid config", asid_cfg_dict)

    
    if env_cfg_dict is not None and sysid and second:
        robot_cfg_dict["model_name"] = robot_cfg_dict["model_name"].replace(
            "base", "second_sysid_" + env_cfg_dict["obj_id"]
        )
    elif env_cfg_dict is not None and sysid:
        robot_cfg_dict["model_name"] = robot_cfg_dict["model_name"].replace(
            "base", "sysid_" + env_cfg_dict["obj_id"]
        )
    elif env_cfg_dict is not None and collision:
        robot_cfg_dict["model_name"] = robot_cfg_dict["model_name"].replace(
            "base", "collision_" + env_cfg_dict["obj_id"]
        )
    elif env_cfg_dict is not None and second:
        robot_cfg_dict["model_name"] = robot_cfg_dict["model_name"].replace(
            "base", "second_" + env_cfg_dict["obj_id"]
        )
    elif env_cfg_dict is not None:
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

            if True or second:
                import numpy as np
                crop_min = np.array([-1, 0, -1])
                crop_max = np.array([1, 1, 1])
                env = ObjectTrackerWrapper(env, **env_cfg_dict, crop_min=crop_min, crop_max=crop_max, verbose=verbose)
            else:
                env = ObjectTrackerWrapper(env, **env_cfg_dict, verbose=verbose)
                
        if asid_cfg_dict is not None:
            from asid.wrapper.asid_wrapper import ASIDWrapper
            env = ASIDWrapper(env, **asid_cfg_dict, verbose=verbose)

            if asid_cfg_dict["reward"]:
                env.create_exp_reward(
                    make_env,
                    robot_cfg_dict,
                    env_cfg_dict,
                    asid_cfg_dict,
                    seed=seed,
                    device_id=device_id,
                )

    env.seed(seed)

    return env


def make_vec_env(
    robot_cfg_dict,
    env_cfg_dict=None,
    asid_cfg_dict=None,
    num_workers=1,
    seed=0,
    device_id=0,
    sysid=False,
    collision=False,
    second=False,
    verbose=False,
):
    from robot.sim.vec_env.vec_wrapper import SubVecEnv

    env_fns = [
        partial(
            make_env,
            robot_cfg_dict,
            env_cfg_dict,
            asid_cfg_dict,
            seed=seed + i,
            device_id=device_id,
            sysid=sysid,
            collision=collision,
            second=second,
            verbose=bool(i == 0) and verbose,
        )
        for i in range(num_workers)
    ]
    return SubVecEnv(env_fns)
