import time
import datetime
import imageio
import numpy as np

from robot.robot_env import RobotEnv

horizon=20
DoF = 6

env = RobotEnv(
    hz=10,
    DoF=DoF,
    robot_model="panda",
    ip_address="localhost", # "172.16.0.1",
    camera_ids=[], # [0,1],
    camera_model="realsense",
    max_lin_vel=1.0,
    max_rot_vel=1.0,
    max_path_length=horizon,
)

from robot.controllers.oculus import VRController
oculus = VRController()

obs = env.reset()
# oculus.reset_state()

while True:
    pose = env._robot.get_ee_pose()
    gripper = env._robot.get_gripper_position()
    state = {"robot_state": {"cartesian_position": pose, "gripper_position": gripper}}
    action, info = oculus.forward(state, include_info=True)
    
    if DoF == 3:
        action = np.concatenate((action[:3], action[-1:]))
    elif DoF == 4:
        action = np.concatenate((action[:3], action[5:6], action[-1:]))

    env.step(action)


# img = obs["img_obs_0"]
# imgs = [img]
acts = []

for i in range(horizon):
    start = time.time()
    # act = - np.array([0., 1.0, 0., -1 if i%2 else 1])
    act = np.zeros(4)
    act[1] = 1.0
    # print(act, time.time() - start)
    obs, reward, done, _ = env.step(act)
    # imgs.append(img)
    acts.append(act)
    # img = obs["img_obs_0"]

# imageio.mimsave('debug.gif', imgs)