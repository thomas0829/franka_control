import time

import hydra
import imageio
import numpy as np
import torch

from asid.wrapper.asid_vec import make_env
from utils.experiment import hydra_to_dict, set_random_seed
from utils.transformations_mujoco import quat_to_euler_mujoco

def compute_cartesian_error(ee_pose, target_pose):
    return np.linalg.norm(ee_pose[:3] - target_pose[:3])

def move_to_cartesian_pose(target_pose, env, error_thresh, max_iter=50):
    ctr = 0
    while True:
        robot_state, _ = env.unwrapped._robot.get_robot_state()
        q_desired = (
            env.unwrapped._robot._ik_solver.cartesian_position_to_joint_position(
                target_pose[:3], target_pose[3:], robot_state
            )
        )
        env.unwrapped._robot.update_joints(q_desired, blocking=True)
        error = compute_cartesian_error(env.unwrapped._robot.get_ee_pose(), target_pose)

        ctr += 1
        if error < error_thresh or ctr > max_iter:
            break

        print("iter", ctr, "error", error)


def move_to_cartesian_pose_delta(
    target_pose,
    gripper,
    env,
    max_iter=None,
    error_thresh=None,
    noise_std=1e-2,
    render=False,
    verbose=False,
):

    imgs = []
    curr_iter = 0
    while True:

        curr_pose = env.unwrapped._robot.get_ee_pose()

        act = np.zeros(7)
        act[:6] = target_pose[:6] - curr_pose[:6]
        act = apply_noise(act, mean=0.0, std=noise_std)
        act[3:6] = np.arctan2(np.sin(act[3:6]), np.cos(act[3:6]))
        act[-1] = gripper

        # cast in case RLDS env_logger is used
        env.step(act.astype(np.float32))
        if render:
            imgs.append(env.render())

        curr_iter += 1
        error = compute_cartesian_error(env.unwrapped._robot.get_ee_pose(), target_pose)
        if verbose:
            print("iter", curr_iter, "error", error, "act", act)

        if (error_thresh is not None and error < error_thresh) or (
            max_iter is not None and curr_iter == max_iter
        ):
            break

    return imgs


def apply_noise(act, mean=0.0, std=1e-1):
    return act.copy() + np.random.normal(loc=mean, scale=std, size=act.shape)


def collect_demo_pick_up(
    env, z_waypoints=[0.3, 0.2, 0.12], noise_std=[5e-2, 1e-2, 5e-3], error_thresh=5e-2, render=False, verbose=False,
):
    """
    Collect a "pick up the red block" demo

    Args:
    - env: robot environment
    - z_waypoints: list of z waypoints for the pick up -> above, closer, down
    - noise_std: list of noise std for each waypoint
    - render: whether to render the environment

    Returns:
    - success: whether the pick up was successful
    - imgs: list of images for each action (empty if render=False)
    """

    imgs = []

    env.reset()

    # get initial target pose
    target_pose = np.zeros(6)
    target_pose[:3] = env.get_obj_pose().copy()[:3]
    target_pose[3:] = quat_to_euler_mujoco(env.get_obj_pose().copy()[3:])
    # overwrite x,y angle w/ gripper default
    target_pose[3:5] = env.unwrapped._default_angle[:2]
    # randomize grasp angle
    rng = np.random.randint(0, 3)
    target_pose[5:] += np.pi / 2 if rng == +np.pi / 2 else -np.pi / 2 if rng == 1 else 0

    # WARNING: real robot EE is offset by 90 deg -> target_pose[5] += np.pi / 4

    # MOVE ABOVE
    target_pose[2] = z_waypoints[0]
    # add pose noise
    target_pose_noise = apply_noise(target_pose, mean=0.0, std=noise_std[0])
    gripper = 0.0
    # move to target pose + add action noise
    imgs.append(
        move_to_cartesian_pose_delta(
            target_pose_noise,
            gripper,
            env,
            error_thresh=error_thresh,
            noise_std=noise_std[0],
            render=render,
            verbose=verbose,
        )
    )

    # MOVE CLOSER
    target_pose[2] = z_waypoints[1]
    target_pose_noise = apply_noise(target_pose, mean=0.0, std=noise_std[1])
    gripper = 0.0
    imgs.append(
        move_to_cartesian_pose_delta(
            target_pose_noise,
            gripper,
            env,
            error_thresh=error_thresh,
            noise_std=noise_std[1],
            render=render,
            verbose=verbose,
        )
    )

    # MOVE DOWN
    target_pose[2] = z_waypoints[2]
    gripper = 0.0
    imgs.append(
        move_to_cartesian_pose_delta(
            target_pose,
            gripper,
            env,
            error_thresh=error_thresh,
            noise_std=noise_std[2],
            render=render,
            verbose=verbose,
        )
    )

    # GRASP
    gripper = 1.0
    imgs.append(
        move_to_cartesian_pose_delta(
            target_pose, gripper, env, max_iter=5, noise_std=noise_std[2], render=render,verbose=verbose,
        )
    )

    # MOVE UP
    target_pose[2] = z_waypoints[1]
    target_pose_noise = apply_noise(target_pose, mean=0.0, std=noise_std[1])
    gripper = 1.0
    imgs.append(
        move_to_cartesian_pose_delta(
            target_pose,
            gripper,
            env,
            error_thresh=error_thresh,
            noise_std=noise_std[1],
            render=render,
            verbose=verbose,
        )
    )

    # MOVE ABOVE
    target_pose[2] = z_waypoints[0]
    target_pose_noise = apply_noise(target_pose, mean=0.0, std=noise_std[1])
    imgs.append(
        move_to_cartesian_pose_delta(
            target_pose,
            gripper,
            env,
            error_thresh=error_thresh,
            noise_std=noise_std[0],
            render=render,
            verbose=verbose,
        )
    )

    success = env.get_obj_pose()[2] > 0.1
    return success, imgs


@hydra.main(config_path="../configs/", config_name="collect_sim", version_base="1.1")
def run_experiment(cfg):

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    set_random_seed(cfg.seed)

    cfg.robot.on_screen_rendering = False
    cfg.env.obj_pos_noise = True

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
    )

    import time

    start = time.time()
    for i in range(100):

        success, imgs = collect_demo_pick_up(
            env,
            z_waypoints=[0.3, 0.2, 0.12],
            noise_std=[5e-2, 1e-2, 5e-3],
            render=False,
        )

        print("success", success, "len", len(imgs))

    # imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)
    print("time", time.time() - start)


if __name__ == "__main__":
    run_experiment()
