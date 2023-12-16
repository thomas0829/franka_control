import time
import datetime
import imageio
import numpy as np

from robot.robot_env import RobotEnv

horizon=20
DoF = 6

env = RobotEnv(
    control_hz=10,
    DoF=DoF,
    robot_type="panda",
    ip_address="localhost", # "172.16.0.1",
    camera_ids=[], # [0,1],
    camera_model="realsense",
    max_lin_vel=1.0,
    max_rot_vel=1.0,
    max_path_length=horizon,
)

obs = env.reset()

# RANDOM MOTION
env.step(np.ones(7)*0.1)
env.step(np.ones(7)*0.1)
env.step(np.ones(7)*0.1)

env.step(-np.ones(7)*0.1)
env.step(-np.ones(7)*0.1)
env.step(-np.ones(7)*0.1)

## OCULUS

# from robot.controllers.oculus import VRController
# oculus = VRController()

# while True:
#     pose = env._robot.get_ee_pose()
#     gripper = env._robot.get_gripper_position()
#     state = {"robot_state": {"cartesian_position": pose, "gripper_position": gripper}}
#     action, info = oculus.forward(state, include_info=True)
    
#     if DoF == 3:
#         action = np.concatenate((action[:3], action[-1:]))
#     elif DoF == 4:
#         action = np.concatenate((action[:3], action[5:6], action[-1:]))

#     env.step(action)