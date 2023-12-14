import time
import datetime
import imageio
import numpy as np

from robot.robot_env import RobotEnv

horizon=20

env = RobotEnv(
    hz=10,
    DoF=3,
    robot_model="panda",
    ip_address="localhost", # "172.16.0.1",
    camera_ids=[], # [0,1],
    camera_model="realsense",
    max_lin_vel=0.5,
    max_rot_vel=1.0,
    max_path_length=horizon,
)

from robot.controllers.oculus import VRController
oculus = VRController()

obs = env.reset()

while True:
    # time.sleep(0.3)
    pos, angle = env._robot.get_ee_pose()
    gripper = env._robot.get_gripper_position()
    state = {"robot_state": {"cartesian_position": np.concatenate((pos, angle)), "gripper_position": gripper}}
    print(oculus.get_info())
    deltas, info = oculus.forward(state, include_info=True)
    act = np.concatenate((deltas[:3], deltas[-1:]))
    print(act)
    env.step(act)


# img = obs["img_obs_0"]
# imgs = [img]
acts = []

for i in range(horizon):
    start = time.time()
    act = - np.array([0., 1.0, 0., -1 if i%2 else 1])
    # print(act, time.time() - start)
    obs, reward, done, _ = env.step(act)
    # imgs.append(img)
    acts.append(act)
    # img = obs["img_obs_0"]

# imageio.mimsave('debug.gif', imgs)