import time

from robot.real.franka_r2d2 import FrankaRobot
from helpers.transformations_r2d2 import *

robot = FrankaRobot(ip_address="localhost")

from robot.controllers.oculus import VRController
oculus = VRController()

home = robot._robot.home_pose
robot.update_command(np.concatenate((home, np.zeros(1))), action_space="joint_position", blocking=True)

while True:
    time.sleep(0.1)
    pose = robot.get_ee_pose()
    gripper = robot.get_gripper_position()
    state = {"robot_state": {"cartesian_position": pose, "gripper_position": gripper}}
    action, info = oculus.forward(state, include_info=True)
    print(action)
    robot.update_command(action, action_space="cartesian_velocity", blocking=False)


euler_move = euler.copy()
euler_move[2] += 0.1

robot.update_pose(pos=pos, angle=euler_move)