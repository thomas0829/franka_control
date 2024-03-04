import argparse
import os
from multiprocessing import Process, Queue

import hydra
import imageio
import joblib
import mujoco
import numpy as np
import torch
from tqdm import trange

from asid.sysid.identifier import SysIdentifier
from asid.wrapper.asid_vec import make_env, make_vec_env
from robot.controllers.cartesian_pd import CartesianPDController
# from robot.controllers.motion_planner import MotionPlanner
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Video, configure_logger
from utils.system import get_device, set_gpu_mode
from utils.transformations import *
from utils.transformations_mujoco import (euler_to_quat_mujoco,
                                          quat_to_euler_mujoco)


def move_to_cartesian_pose(
    target_pose,
    gripper,
    controller,
    env,
    progress_threshold=1e-3,
    max_iter_per_waypoint=20,
    render=False,
    verbose=False,
):

    return jump_to_cartesian_pose(
        target_pose,
        gripper,
        env,
        render=render,
        verbose=verbose,
    )

    controller.reset()

    # start = env.unwrapped._robot.get_joint_positions().copy()
    start = env.unwrapped._robot.get_ee_pose().copy()
    start = np.concatenate((start[:3], euler_to_quat_mujoco(start[3:])))
    target_pose = target_pose.copy()

    if target_pose[5] > np.pi / 2:
        target_pose[5] -= np.pi
    if target_pose[5] < -np.pi / 2:
        target_pose[5] += np.pi

    goal = np.concatenate((target_pose[:3], euler_to_quat_mujoco(target_pose[3:])))
    # qpos_plan = motion_planner.plan_motion(start, goal, return_ee_pose=True)

    steps = 0
    error = []
    imgs = []

    # for i in range(len(qpos_plan.ee_position)):
    #     des_pose = np.concatenate(
    #         (
    #             qpos_plan.ee_position[i].cpu().numpy(),
    #             quat_to_euler_mujoco(qpos_plan.ee_quaternion[i].cpu().numpy()),
    #         )
    #     )
    for i in range(1):

        des_pose = target_pose
        # des_pose[5] = des_pose[5] / 2 # scale to np.pi/2

        # print("des_pose", des_pose)
        last_curr_pose = des_pose

        for j in range(max_iter_per_waypoint):

            # get current pose
            curr_pose = env.unwrapped._robot.get_ee_pose()

            # run PD controller
            act = controller.update(curr_pose, des_pose)
            # act[3:] = np.arctan2(np.sin(act[3:] * 2), np.cos(act[3:] * 2)) / 2
            act = np.concatenate((act, [gripper]))

            # print("angle act", act[3:], "euler", curr_pose[3:])
            # print("angle act", act[3:], "euler", curr_pose[3:])
            # step env
            obs, _, _, _ = env.step(act)
            steps += 1

            # compute error
            if verbose:
                curr_pose = env.unwrapped._robot.get_ee_pose()
                err_pos = np.linalg.norm(target_pose[:3] - curr_pose[:3])
                err_angle = np.linalg.norm(target_pose[3:] - curr_pose[3:])
                err = err_pos  # + err_angle
                error.append(err)

                print(
                    j,
                    "err",
                    err_pos,
                    err_angle,
                    "pose norm",
                    np.linalg.norm(last_curr_pose - curr_pose),
                )  # "act_max_abs", np.max(np.abs(act)), "act", act)

            if render:
                imgs.append(env.render())
            env.render()

            # early stopping when actions don't change position anymore
            if np.linalg.norm(des_pose[:3] - curr_pose[:3]) < progress_threshold:
                break
            last_curr_pose = curr_pose

    return imgs, steps


def jump_to_cartesian_pose(
    target_pose,
    gripper,
    env,
    render=False,
    verbose=False,
):

    target_pose = target_pose.copy()

    if target_pose[5] > np.pi / 2:
        target_pose[5] -= np.pi
    if target_pose[5] < -np.pi / 2:
        target_pose[5] += np.pi

    steps = 0

    if gripper:
        env.unwrapped._robot.update_gripper(gripper)

    while np.linalg.norm(target_pose[:3] - env.unwrapped._robot.get_ee_pose()[:3]) > 5e-2:
    # while np.linalg.norm(target_pose - env.unwrapped._robot.get_ee_pose()) > 5e-2:
        robot_state = env.unwrapped._robot.get_robot_state()[0]
        desired_qpos = (
            env.unwrapped._robot._ik_solver.cartesian_position_to_joint_position(
                target_pose[:3], euler_to_quat(target_pose[3:]), robot_state
            )
        )
        # env.unwrapped._robot.move_to_joint_positions(desired_qpos)
        env.unwrapped._robot.update_desired_joint_positions(desired_qpos.tolist())
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


def collect_rollout(
    env, action, control_hz=10, render=False, verbose=False
):

    env.reset()

    rod_pose = env.get_obj_pose()

    controller = CartesianPDController(Kp=0.7, Kd=0.0, control_hz=control_hz)
    imgs = []

    progress_threshold = 0.1

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
    tmp, _ = move_to_cartesian_pose(
        target_pose,
        gripper,
        controller,
        env,
        progress_threshold=progress_threshold,
        max_iter_per_waypoint=20,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE DOWN
    target_pose[2] = 0.2
    gripper = 0.0
    tmp, _ = move_to_cartesian_pose(
        target_pose,
        gripper,
        controller,
        env,
        progress_threshold=progress_threshold,
        max_iter_per_waypoint=20,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE DOWN
    # go down manually -> better than MP
    while env.unwrapped._robot.get_ee_pose()[2] > 0.13 or (
        env.unwrapped.sim and env.unwrapped._robot.get_ee_pose()[2] > 0.113
    ):
        env.step(np.array([0.0, 0.0, -0.03, 0.0, 0.0, 0.0, 0.0]))

    # GRASP
    # make sure gripper is fully closed -> 3 steps
    for _ in range(3):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    # MOVE UP
    target_pose[2] = 0.2
    gripper = 1.0
    tmp, _ = move_to_cartesian_pose(
        target_pose,
        gripper,
        controller,
        env,
        progress_threshold=progress_threshold,
        max_iter_per_waypoint=20,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE UP
    target_pose[2] = 0.3
    gripper = 1.0
    tmp, _ = move_to_cartesian_pose(
        target_pose,
        gripper,
        controller,
        env,
        progress_threshold=progress_threshold,
        max_iter_per_waypoint=20,
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
    tmp, _ = move_to_cartesian_pose(
        target_pose,
        gripper,
        controller,
        env,
        progress_threshold=progress_threshold,
        max_iter_per_waypoint=20,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    # MOVE DOWN
    target_pose[2] = 0.15
    gripper = 1.0
    tmp, _ = move_to_cartesian_pose(
        target_pose,
        gripper,
        controller,
        env,
        progress_threshold=progress_threshold,
        max_iter_per_waypoint=20,
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
    tmp, _ = move_to_cartesian_pose(
        target_pose,
        gripper,
        controller,
        env,
        progress_threshold=progress_threshold,
        max_iter_per_waypoint=20,
        render=render,
        verbose=verbose,
    )
    imgs += tmp

    rod_orn = quat_to_euler_mujoco(env.get_obj_pose()[-4:])
    # reward = -((rod_orn[0] - init_rod_pitch) ** 2)
    reward = -np.linalg.norm(rod_orn[0] - init_rod_pitch)
    return reward, imgs


reset_joint_qpos = np.array(
    [
        0.00337199,
        -0.02690877,
        -0.00988887,
        -2.33686253,
        0.01294934,
        2.35197591,
        0.11204342,
    ]
)


def train_cem_policy(cfg, zeta=None):

    num_iters, num_samples, num_procs = (
        cfg.train.algorithm.num_iters,
        cfg.train.algorithm.num_samples,
        cfg.train.algorithm.num_workers,
    )
    num_elites = int(cfg.train.algorithm.elite_frac * cfg.train.algorithm.num_samples)

    action_mean = cfg.train.algorithm.mu_init
    action_std = cfg.train.algorithm.sigma_init

    motion_planner = None
    # motion_planner = MotionPlanner(interpolation_dt=0.1, device=torch.device("cuda:0"))

    q = Queue()
    batch_size = num_samples // num_procs
    for _ in trange(num_iters, desc="CEM iteration"):

        # for i in  trange(num_samples // batch_size, desc="CEM batch", leave=False):

        procs = [
            Process(target=cem_rollout_worker, args=(q, cfg, batch_size, action_mean, action_std, zeta))
            for _ in range(num_procs)
        ]
            
        # actions, rewards = cem_rollout_worker(
        #     q,
        #     cfg,
        #     batch_size,
        #     action_mean,
        #     action_std,
        #     zeta,
        #     motion_planner,
        #     verbose=False,
        #     render=False,
        # )

        for p in procs:
            p.start()

        for p in procs:
            p.join()

        # for i in trange(num_samples):
        # procs = [
        #     Process(target=cem_rollout_worker, args=(q, cfg, batch_size, action_mean, action_std, zeta, motion_planner))
        #     for _ in range(num_procs)
        # ]

        # for p in procs:
        #     p.start()

        # for p in procs:
        #     p.join()

        # Update mean and std
        results = [q.get() for _ in range(num_procs)]
        actions = np.concatenate([res["actions"] for res in results], axis=0)
        rewards = np.concatenate([res["rewards"] for res in results], axis=0)

        elites = actions[np.cfgort(rewards)][-num_elites:]
        action_mean = np.mean(elites, axis=0)
        action_std = np.std(elites, axis=0)
        print(
            f"action_mean: {action_mean}, action_std: {action_std}, reward: {np.mean(rewards)} zeta: {zeta}"
        )
    return action_mean, action_std


def cem_rollout_worker(
    q,
    cfg,
    num_rollouts,
    action_mean,
    action_std,
    zeta,
    verbose=False,
    render=False,
):

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed,
        device_id=cfg.gpu_id,
        collision=True,
    )
    env.reset()

    actions = np.zeros((num_rollouts), dtype=np.float32)
    rewards = np.zeros((num_rollouts), dtype=np.float32)

    for i in range(num_rollouts):

        if zeta is not None:
            env.set_parameters(zeta)
        
        # Sample action
        action = np.random.normal(action_mean, action_std)

        # if motion_planner is None:
        #     motion_planner = MotionPlanner(
        #         interpolation_dt=0.1, device=torch.device("cuda:0")
        #     )

        # Collect rollout
        reward, _ = collect_rollout(
            env,
            action,
            control_hz=cfg.robot.control_hz,
            verbose=verbose,
            render=render,
        )

        actions[i] = action
        rewards[i] = reward

    return actions, rewards
    # q.put({"actions": actions, "rewards": rewards})


# def collect_rollout(env, action, log_video=False, video_path=""):
#     # Scale action within (-0.1, 0.1)
#     action = np.clip(action, -1, 1) * 0.1

#     # Ensure frame skip is 20
#     env.frame_skip = 20

#     # Reset environment without changing parameters
#     for _ in range(5):
#         env.update_joints(reset_joint_qpos)
#         curr_joints = env.get_joint_positions()
#         joint_dist = np.linalg.norm(curr_joints - reset_joint_qpos)
#         if joint_dist < 0.1:
#             break
#     env.update_rod(env.init_rod_pose.copy())

#     if log_video:
#         frames = []

#     # Step 1: use ik to align franka above the rod
#     rod_pose = env.get_rod_pose()
#     target_pos = rod_pose[:3]
#     target_pos[2] += 0.2

#     target_orn = quat_to_euler_mujoco(rod_pose[-4:])
#     init_rod_pitch = target_orn[0]
#     init_rod_yaw = target_orn[2]
#     target_orn[0] -= np.pi
#     target_orn[2] += np.pi / 2

#     # Set grasp target to center of mass
#     com = action
#     target_pos[0] -= com * np.sin(init_rod_yaw)
#     target_pos[1] += com * np.cos(init_rod_yaw)

#     for _ in range(20):
#         env.update_pose(target_pos, target_orn, 255)
#         if log_video:
#             frames.append(env.render(mode="rgb_array")[0]["color_image"])

#     # Step 2: lower the franka
#     target_pos[2] -= 0.1
#     for _ in range(10):
#         env.update_pose(target_pos, target_orn, 200)
#         if log_video:
#             frames.append(env.render(mode="rgb_array")[0]["color_image"])

#     # Step 3: close the gripper
#     for _ in range(10):
#         env.update_pose(target_pos, target_orn, 0)
#         if log_video:
#             frames.append(env.render(mode="rgb_array")[0]["color_image"])

#     # Step 4: lift the rod
#     stand_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "stand")
#     target_pos = env.model.geom_pos[stand_geom_id].copy()
#     target_pos[2] += 0.25

#     target_orn = quat_to_euler_mujoco(env.model.geom_quat[stand_geom_id].copy())
#     target_orn[0] -= np.pi
#     target_orn[2] += np.pi / 2

#     for _ in range(20):
#         env.update_pose(target_pos, target_orn, 0)
#         if log_video:
#             frames.append(env.render(mode="rgb_array")[0]["color_image"])

#     # Step 5: open the gripper
#     for _ in range(10):
#         env.update_pose(target_pos, target_orn, 255)
#         if log_video:
#             frames.append(env.render(mode="rgb_array")[0]["color_image"])

#     # Step 6: let the rod fall
#     target_pos[2] += 0.2
#     for _ in range(50):
#         env.update_pose(target_pos, target_orn, 255)
#         if log_video:
#             frames.append(env.render(mode="rgb_array")[0]["color_image"])

#     # Evaluate final rod angle
#     rod_orn = quat_to_euler_mujoco(env.get_rod_pose()[-4:])
#     reward = -(rod_orn[0] - init_rod_pitch) ** 2

#     if log_video:
#         video = np.stack(frames, axis=0)
#         imageio.mimwrite(os.path.join(video_path), video)#, fps=15)

#     return reward


@hydra.main(config_path="../configs/", config_name="task_rod_sim", version_base="1.1")
def run_experiment(cfg):
    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[explore][{cfg.seed}]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    set_random_seed(cfg.seed)
    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "explore")
    logger = configure_logger(logdir, cfg.log.format_strings)

    # cfg.robot.on_screen_rendering = True
    cfg.robot.gripper = True
    cfg.asid.obs_noise = 0.0

    # Load zeta parameter
    zeta_dir = os.path.join(logdir, cfg.exp_id, str(cfg.seed), "sysid", "zeta")
    if os.path.exists(zeta_dir):
        zeta_dict = joblib.load(zeta_dir)
        for k, v in zeta_dict.items():
            zeta_dict[k] = np.array(v)
    else:
        print("Using default zeta_dict")
        zeta_dict = {"mu": np.array([0.07]), "": np.array([0.08])}

    # Train policy
    if cfg.train.mode == "manual":
        action = cfg.train.action
    
    else:
        if cfg.train.mode == "sysid":
            action_mean, action_std = train_cem_policy(cfg, zeta_dict["mu"])
        elif cfg.train.mode == "domain_rand":
            action_mean, action_std = train_cem_policy(cfg, None)
        
        action = np.random.normal(action_mean, action_std)
        print(
                f"{cfg.exp_id} {cfg.train.mode} action_mean: {action_mean}, action_std: {action_std}"
            )
        
        param_dict = {
            "mu": action_mean,
            "sigma": action_std,
        }
        joblib.dump(param_dict, os.path.join(logdir, cfg.exp_id, str(cfg.seed), "task", "policy"))
        
    # Evaluate policy
    eval_env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed + 100,
        device_id=cfg.gpu_id,
        collision=True,
    )
    eval_env.set_parameters(zeta_dict[""])
    video_path = os.path.join(cfg.logdir, cfg.exp_id, str(cfg.seed), "sysid")
    os.makedirs(video_path, exist_ok=True)
    reward = collect_rollout(
        eval_env,
        action,
        log_video=True,
        video_path=os.path.join(video_path, f"{cfg.train.mode}.gif"),
    )
    print(
        f"FINAL real zeta {zeta_dict['']} EXP {cfg.exp_id} ALGO {cfg.train.mode} reward: {reward} act {action} act_mean {action_mean} act_std {action_std}"
    )


if __name__ == "__main__":
    run_experiment()
