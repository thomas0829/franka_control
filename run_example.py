
import os
import time
import datetime
import imageio
import numpy as np

from robot.robot_env import RobotEnv
from helpers.discretization import inverse_discretize

horizon=20

env = RobotEnv(
    hz=10,
    DoF=3,
    robot_model="panda",
    ip_address="172.16.0.1",
    camera_ids=[0],
    camera_model="realsense",
    max_lin_vel=0.05,
    max_rot_vel=1.0,
    max_path_length=horizon,
)

obs = env.reset()
img = obs["imgs_obs_0"]
imgs = [img]
acts = []

for i in range(horizon):
    start = time.time()
    act = - np.array([0., 1.0, 0., 0.])
    print(act, time.time() - start)
    obs, reward, done, _ = env.step(act)
    imgs.append(img)
    acts.append(act)
    img = obs["imgs_obs_0"]

date_str = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
os.makedirs("results", exist_ok=True)
imageio.mimsave(f'results/rollout-{date_str}.gif', imgs)