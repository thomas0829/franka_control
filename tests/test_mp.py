import time

import hydra
import numpy as np
import torch

from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict
from utils.transformations import euler_to_quat, quat_to_euler
from utils.transformations_mujoco import *


def wrap_angle(angle_diff):
    return (angle_diff + np.pi) % (2 * np.pi) - np.pi

def scale_angles(euler):
    return np.arctan2(np.sin(euler), np.cos(euler))

class PDController:
    def __init__(self, Kp, Kd, control_hz=10):
        self.Kp = Kp  # Proportional gain
        self.Kd = Kd  # Derivative gain
        self.pos_prev_error = 0
        self.quat_prev_error = 0
        self.dt = 1/control_hz

    def update(self, curr, des):
        """
        Update the PD controller.

        Args:
            des (float): The desired value.
            curr (float): The current value.

        Returns:
            float: The control output.
        """
        # Calculate the position error
        pos_error = des[:3] - curr[:3]

        # Calculate the derivative of the position error
        pos_error_dot = (pos_error - self.pos_prev_error) / self.dt

        # Update the previous position error and time for the next iteration
        self.pos_prev_error = pos_error

        # Calculate the position control output
        u_pos = self.Kp * pos_error + self.Kd * pos_error_dot


        # Calculate the quaternion error
        # quat_error = subtract_euler_mujoco(des[3:], curr[3:])
        quat_error = des[3:] - curr[3:]
        quat_error = np.arctan2(np.sin(quat_error), np.cos(quat_error))

        # Calculate the derivative of the quaternion error
        quat_error_dot = (quat_error - self.quat_prev_error) / self.dt

        # Update the previous quaternion error and time for the next iteration
        self.quat_prev_error = quat_error

        # Calculate the quaternion control output
        u_quat = self.Kp * quat_error + self.Kd * quat_error_dot
        

        # Combine the position and quaternion control outputs
        u = np.concatenate((u_pos, u_quat))

        return u


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

    cfg.robot.on_screen_rendering = False
    fk_env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=False,
    )
    fk_env.reset()
    
    from robot.controllers.motion_planner import MotionPlanner
    motion_planner = MotionPlanner(interpolation_dt=0.7, device=torch.device("cuda:0"))

    qpos = env.unwrapped._robot.get_joint_positions()
    start = qpos

    ee_pose = env.unwrapped._robot.get_ee_pose()
    goal = ee_pose.copy()
    goal[:3] = goal[:3] + np.array([-0.1, 0.2, 0.1])
    goal[3:] = goal[3:] + np.array([0.0, 0.2, 0.5])
    goal_pose = np.concatenate((goal[:3], euler_to_quat_mujoco(goal[3:])))

    qpos_plan = motion_planner.plan_motion(start, goal_pose, return_ee_pose=True)

    # # verify plan
    # fk_env._robot.update_desired_joint_positions(joint_pos_desired=qpos_plan[-1].position.cpu())
    # des_pose = fk_env._robot.get_ee_pose()

    des_pose = np.concatenate((qpos_plan.ee_position[-1].cpu().numpy(), quat_to_euler_mujoco(qpos_plan.ee_quaternion[-1].cpu().numpy())))
    goal_pose_tmp = goal_pose.copy()
    goal_quat_tmp = quat_to_euler_mujoco(goal_pose[3:])
    goal_pose_tmp = np.concatenate((goal_pose[:3], goal_quat_tmp))
    print("goal - traj[-1]:", des_pose - goal_pose_tmp)

    pd_euler = PDController(Kp=1., Kd=0.)

    error = []
    progress_threshold = 1e-3
    max_iter_per_waypoint = 20
    steps = 0

    for i in range(len(qpos_plan.ee_position)):
    # for i in range(1):
        
        # fk_env._robot.update_desired_joint_positions(joint_pos_desired=qpos_plan[i].position.cpu())
        # des_pose = fk_env._robot.get_ee_pose()

        # des_pose = goal

        des_pose = np.concatenate((qpos_plan.ee_position[i].cpu().numpy(), quat_to_euler_mujoco(qpos_plan.ee_quaternion[i].cpu().numpy())))

        last_curr_pose = des_pose

        for j in range(max_iter_per_waypoint):
            
            # get current pose
            curr_pose = env.unwrapped._robot.get_ee_pose()

            # run PD controller
            act = pd_euler.update(curr_pose, des_pose)
            act = np.concatenate((act, np.zeros(1)))
            
            # step env
            obs, _, _, _ = env.step(act)
            steps += 1
            
            # compute error
            curr_pose = env.unwrapped._robot.get_ee_pose()
            err_pos = np.linalg.norm(goal[:3]-curr_pose[:3])
            err_angle = np.linalg.norm(goal[3:] - curr_pose[3:])
            err = err_pos + err_angle
            error.append(err)

            # log
            print(j, "err", err_pos, err_angle, "pose norm", np.linalg.norm(last_curr_pose-curr_pose)) # "act_max_abs", np.max(np.abs(act)), "act", act)
            # print(j, "curr_pose", curr_pose)
            env.render()

            # early stopping when actions don't change position anymore
            if np.linalg.norm(last_curr_pose-curr_pose) < progress_threshold:
                break
            last_curr_pose = curr_pose

    print(f"Reached goal after {steps} steps")

    import matplotlib.pyplot as plt

    # plt.plot(currs)
    # plt.plot(dess, linestyle="--")
    plt.plot(error)
    plt.show()


    env.reset()
    # for i in range(len(plan.ee_position)):
    #     ee_pose = np.concatenate((plan.ee_position[i].cpu().numpy(), euler_to_quat_mujoco(quat_to_euler(plan.ee_quaternion[i].cpu().numpy()))))
    #     env.unwrapped._robot.update_command(ee_pose - env.unwrapped._robot.get_ee_pose(), action_space="cartesian_position", blocking=True)
    #     env.render()
    #     time.sleep(0.1)

    # start = time.time()
    # for i in range(15):
    #     env.reset()
    #     for i in range(5):
    #         obs, reward, done, info = env.step(env.action_space.sample())
    #         env.render()
    # print(time.time() - start)


if __name__ == "__main__":
    run_experiment()


