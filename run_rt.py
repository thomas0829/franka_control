import numpy as np
import matplotlib.pyplot as plt
from robot_env import RobotEnv

import os
import base64
import requests
import time
import datetime

from helpers import inverse_discretize

import imageio
import cv2

horizon=20

def call_rt(msg):
    ip_address = 'https://contributions-provides-bound-spanking.trycloudflare.com'
    # IP address always changes every time the server is rebooted. Please check the text-generation-webui cli interface
    start = time.time()
    with open('obs.png', 'rb') as f:
        img_str = base64.b64encode(f.read()).decode('utf-8')
        prompt = f'### human: \nWhat action should the robot take to `{msg}`\n<img src="data:image/jpeg;base64,{img_str}">### gpt: '
        response = requests.post(f'{ip_address}/v1/completions', json={'prompt': prompt, 'max_tokens': 256, 'stopping_strings': ['\n###']}).json()
        reponse_txt = response['choices'][0]['text']
        minmaxlst = [[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]]

        action = inverse_discretize(reponse_txt, minmaxlst)[:3]
        action = np.concatenate((action, np.ones(1)))

    end = time.time()
    print(f'Took { end - start } seconds')
    return action

env = RobotEnv(
    hz=10, # 1 for RT
    DoF=3,
    robot_model="panda", # "FR3",
    randomize_ee_on_reset=False,
    hand_centric_view=False,
    third_person_view=True,
    ip_address="localhost", # "172.16.0.1",
    local_cameras=True,
    camera_model="realsense",
    max_lin_vel=0.05,
    max_rot_vel=1.0
)

msg = 'go towards the blue block and pick up the blue block'

obs = env.reset()
img = obs["third_person_img_obs"]
imgs = [img]
acts = []

for i in range(horizon):
    obs_resize = cv2.resize(obs["third_person_img_obs"], (336,336))
    plt.imsave("obs.png", obs_resize)
    start = time.time()
    # act = call_rt(msg)
    act = - np.array([0., 1.0, 0., 0.])
    print(act, time.time() - start)
    obs, reward, done, _ = env.step(act)
    imgs.append(img)
    acts.append(act)
    img = obs["third_person_img_obs"]

date_str = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
os.makedirs("results", exist_ok=True)
np.save(f"results/{date_str}-acts", acts)
np.save(f"results/{date_str}-obs", imgs)
np.save(f"results/{date_str}-msg", np.array(msg))
imageio.mimsave(f'rollout-{date_str}.gif', imgs)