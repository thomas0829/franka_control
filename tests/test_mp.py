import time
import torch
import hydra

import numpy as np
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict
from utils.transformations import euler_to_quat, quat_to_euler
from utils.transformations_mujoco import euler_to_quat_mujoco

@hydra.main(config_path="../configs/", config_name="default", version_base="1.1")
def run_experiment(cfg):

    cfg.robot.DoF = 6
    cfg.robot.on_screen_rendering = True
    cfg.robot.gripper = True

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    env.reset()

    from robot.controllers.motion_planner import MotionPlanner
    motion_planner = MotionPlanner(device=torch.device("cuda:0"))

    qpos = env.unwrapped._robot.get_joint_positions()
    start = qpos
    
    ee_pose = env.unwrapped._robot.get_ee_pose()
    goal = np.concatenate((ee_pose[:3], euler_to_quat_mujoco(ee_pose[3:])))
    goal[0] -= 0.1
    goal[1] += 0.2
    goal[2] += 0.1

    plan = motion_planner.plan_motion(start, goal)

    # iterator over plan sometime throws value error -> access by index instead
    for i in range(len(plan)):
        env.unwrapped._robot.update_joints(plan[i].position.cpu().numpy(), blocking=True)
        env.render()
        time.sleep(0.1)

    # TODO debug this!
    # plan = motion_planner.plan_motion(start, goal, return_ee_pose=True)

    # env.reset()
    # for i in range(len(plan.ee_position)):
    #     ee_pose = np.concatenate((plan.ee_position[i].cpu().numpy(), euler_to_quat_mujoco(quat_to_euler(plan.ee_quaternion[i].cpu().numpy()))))
    #     env.unwrapped._robot.update_command(ee_pose, action_space="cartesian_position", blocking=True)
    #     env.render()
    #     time.sleep(0.1)

    start = time.time()
    for i in range(15):
        env.reset()
        for i in range(5):
            obs, reward, done, info = env.step(env.action_space.sample())
            env.render()
    print(time.time() - start)


if __name__ == "__main__":
    run_experiment()
