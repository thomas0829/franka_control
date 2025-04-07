import glob
import os
import time
import h5py
import pickle

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict
#from openrt.scripts.convert_np_to_hdf5 import normalize, unnormalize
from robot.wrappers.crop_wrapper import CropImageWrapper
from robot.wrappers.resize_wrapper import ResizeImageWrapper
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import cv2

# GroundVLA imports
from PIL import Image
import requests
import json_numpy
json_numpy.patch()
import re
# def send_request(image: Image.Image, instruction: str, server_url: str):
#     """
#     Send the captured image and instruction to the inference server using json_numpy.
#     Returns the action output as received from the server.
#     """
#     # Convert PIL image to a NumPy array
#     image_np = np.array(image)
    
#     # Prepare the payload with the image and instruction from the script
#     payload = {
#         "image": image_np,
#         "instruction": instruction
#     }
    
#     headers = {"Content-Type": "application/json"}
#     response = requests.post(server_url, headers=headers, data=json_numpy.dumps(payload))

#     if response.status_code != 200:
#         raise Exception(f"Server error: {response.text}")
#     print("response: ", response)
#     response_data = response.json()

#     import pdb;pdb.set_trace()
#     # action = json_numpy.loads(response_data)
#     # print(action)
#     return response_data

def send_request(image: Image.Image, instruction: str, server_url: str):
    """
    Send the captured image and instruction to the inference server using json_numpy.
    Returns the action output as received from the server.
    """
    # Convert PIL image to a NumPy array
    image_np = np.array(image)
    # Prepare the payload with the image (as 'full_image') and instruction
    payload = {
        "observation": {
             "full_image": image_np
        },
        "instruction": instruction
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(server_url, headers=headers, data=json_numpy.dumps(payload))
    if response.status_code != 200:
        raise Exception(f"Server error: {response.text}")
    print("response: ", response)
    response_data = np.array(response.json())
    print("response_data: ", response_data)
    
    return response_data

def parse_pose_output(output):
    """
    Parse the model's output to extract pose and gripper information.
    Expected to return 8 numbers: 3 for position, 1 for gripper state, and 4 for quaternion orientation.
    If only 7 numbers are provided (i.e. missing one quaternion component), we compute the missing value assuming
    a unit quaternion.
    :param output: The output from the model (can be bytes, string, list, or numpy array).
    :return: A tuple: (position (x, y, z), gripper_state, orientation (x, y, z, w))
    """
    # If output is bytes, decode it.
    if isinstance(output, bytes):
        output = output.decode('utf-8')

    # If output is a list or numpy array, convert to a list of floats.
    if isinstance(output, (list, np.ndarray)):
        numbers = [float(x) for x in output]
    else:
        # Assume it's a string; use regex to find numbers.
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", output)
        numbers = [float(n) for n in numbers]

    if len(numbers) == 7:
        # Assume order: [x, y, z, gripper_state, q_x, q_y, q_z]
        pos = tuple(numbers[0:3])
        gripper_state = numbers[-1]
        q_xyz = numbers[4:7]
        # Compute the missing quaternion component assuming a unit quaternion:
        s = sum(q * q for q in q_xyz)
        # Protect against small negative due to floating point precision.
        missing = max(0.0, 1.0 - s)
        q_w = missing**0.5
        orientation = tuple(q_xyz + [q_w])
        return pos, gripper_state, orientation
    elif len(numbers) == 8:
        pos = tuple(numbers[0:3])
        gripper_state = numbers[3]
        orientation = tuple(numbers[4:8])
        return pos, gripper_state, orientation
    else:
        raise ValueError(f"[Utils] Expected 7 or 8 numbers in predicted action, got: {len(numbers)}")

def shortest_angle(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi
    
def get_dict(demo):
    dic = {}
    obs_keys = demo[0].keys()
    for key in obs_keys:
        dic[key] = np.stack([d[key] for d in demo])
    return dic

# def action_preprocessing(dic, actions):
#         # compute actual deltas s_t+1 - s_t (keep gripper actions)
#     actions_tmp = actions.copy()
#     actions_tmp[:-1, ..., :6] = (
#         dic["lowdim_ee"][1:, ..., :6] - dic["lowdim_ee"][:-1, ..., :6]
#     )
#     actions = actions_tmp[:-1]
    

#         # compute shortest angle -> avoid wrap around
#     actions[..., 3:6] = shortest_angle(actions[..., 3:6])

#     # real data source
#     #actions[..., [3,4,5]] = actions[..., [4,3,5]]
#     #actions[...,4] = -actions[...,4]
#     # actions[...,3] = -actions[...,3] this is a bug

#     print(f'Action min & max: {actions[...,:6].min(), actions[...,:6].max()}')

#     return actions

def action_preprocessing(dic, actions, interval=1):
    """
    Compute action deltas based on a specified interval.
    
    Args:
        dic (dict): Dictionary containing stacked episode data.
        actions (np.array): Original actions of shape (N, 7).
        interval (int): Step size for computing differences (default=1).
    
    Returns:
        np.array: Processed actions of shape (N - interval, 7).
    """
    actions_tmp = actions.copy()

    # Compute deltas with interval steps
    actions_tmp[:-interval, ..., :6] = (
        dic["lowdim_ee"][interval:, ..., :6] - dic["lowdim_ee"][:-interval, ..., :6]
    )

    # Keep only actions that have valid deltas
    actions = actions_tmp[:-interval]

    # Compute shortest angle to avoid wrap-around issues
    actions[..., 3:6] = shortest_angle(actions[..., 3:6])

    print(f'Action min & max: {actions[..., :6].min(), actions[..., :6].max()}')

    return actions


    
def replay_episode(demo, env, length=-1, interval=1, visual=False):
    # stack data
    dic = get_dict(demo)
    actions = np.stack([d["action"] for d in demo])
    actions = action_preprocessing(dic, actions, interval=interval) # delta action
    demo_length = actions.shape[0]

    if length == -1:
        length = demo_length
    else:
        length = length

    for step_idx in tqdm(range(0, length-1)):
        act = actions[step_idx]

        obs = env.get_observation()

        if visual:
            cv2.imshow("Camera View", obs["215122252864_rgb"])
            cv2.waitKey(1)  # Small delay to allow the image to refresh
        
        next_obs, reward, done, info = env.step(act)
        # print("action: ", act)

    cv2.destroyAllWindows()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)

# config_name="collect_demos_real" for real robot config_name="collect_demos_sim" for simulation
@hydra.main(
    # config_path="../../configs/", config_name="collect_demos_real", version_base="1.1"
    config_path="../../configs/", config_name="eval_openvla_real", version_base="1.1"
)
def run_experiment(cfg):

    # logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    # os.makedirs(logdir, exist_ok=True)

    set_random_seed(cfg.seed)
    assert cfg.robot.imgs, "ERROR: set robot.imgs=true to record image observations!"
    cfg.robot.max_path_length = cfg.max_episode_length
    print("[INFO]", cfg.robot.blocking_control, cfg.robot.control_hz)
    #assert cfg.robot.blocking_control==True and cfg.robot.control_hz<=1, "WARNING: please make sure to pass robot.blocking_control=true robot.control_hz=1 to run blocking control!"
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env) if "env" in cfg.keys() else None,
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    camera_names = [k for k in env.get_images().keys()]

    print(f"Camera names: {camera_names}")

    # crop image observations
    if cfg.aug.camera_crop is not None:
        env = CropImageWrapper(
            env,
            x_min=cfg.aug.camera_crop[0],
            x_max=cfg.aug.camera_crop[1],
            y_min=cfg.aug.camera_crop[2],
            y_max=cfg.aug.camera_crop[3],
            image_keys=[cn + "_rgb" for cn in camera_names],
            crop_render=True,
        )

    # resize image observations
    '''
    if cfg.aug.camera_resize is not None:
        env = ResizeImageWrapper(
            env,
            size=cfg.aug.camera_resize,
            image_keys=[cn for cn in camera_names],
        )
    '''

    def image_preprocessing(image):
        image = cv2.resize(image, [360, 360])
        return image
    
    env.seed(cfg.seed)
    obs = env.reset()


    

    mode = "close_loop" # replay, close_loop, open_loop
    # demo_dir = "/home/prior/data_collection/data/38/pick_duster_npy/pick_duster_1/train/episode_2.npy"
    demo_dir = "/home/prior/rlds_dataset_builder/pick_mango/data/pick_mango_1/train/episode_2.npy"

    url = cfg.url
    print("[INFO] mode: ", mode)
    print("[INFO] url: ", url)
    if mode == "replay":
        print('[INFO] demo_dir: ', demo_dir)
        demo = np.load(demo_dir, allow_pickle=True)
        replay_episode(demo, env, interval=1, visual=True)
    elif mode == "close_loop":
        for i in range(cfg.traj_length): 
            print(f'Step: {i} | lang: {cfg.msg}')
            start = time.time()
            img = image_preprocessing(obs[camera_names[0]+"_rgb"])
            msg = cfg.msg

            # sanity check            
            # from PIL import Image
            # Image.fromarray(img).save("/home/prior/tmp/groundvla_eval.png")
            # print("Image saved as /home/prior/tmp/groundvla_eval.png")

            # ground vla inference
            act = send_request(img, msg, url)
            print(f"[OpenVLA Client] Received Action (deltas): {act}")
            
            # step
            next_obs, reward, done, info = env.step(act)
            print(
                f"Time {np.around(time.time()-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(act,3)}"
            )
            obs = next_obs

    elif mode == "open_loop":
        print('[INFO] demo_dir: ', demo_dir)
        demo = np.load(demo_dir, allow_pickle=True)
        dic = get_dict(demo)
        
        inference_mean = []
        for i in range(len(demo)):
            print(f'Step: {i} | lang: {cfg.msg}')
            start = time.time()
            img = image_preprocessing(dic[camera_names[0]+"_rgb"][i])
            msg = cfg.msg

            # ground vla inference
            act = send_request(img, msg, url)
            # print(f"[OpenVLA Client] Received Action (deltas): {act}")
            
            # step
            _, _, _, _ = env.step(act)
            # print(
            #     f"Time {np.around(time.time()-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)}"
            # )
            inference_mean.append(act)
            print(
                f"Time {np.around(time.time()-start, 3)} | Mean: {np.array(inference_mean).mean()} | Act {act}"
            )

    elif mode == "mix":
        print('[INFO] demo_dir: ', demo_dir)
        demo = np.load(demo_dir, allow_pickle=True)
        replay_episode(demo, env, length=15)

        # get obs again
        obs = env.get_observation()
        for i in range(cfg.traj_length): 
            print(f'Step: {i} | lang: {cfg.msg}')
            start = time.time()
            img = image_preprocessing(obs[camera_names[0]+"_rgb"])
            msg = cfg.msg

            # sanity check            
            # from PIL import Image
            # Image.fromarray(img).save("/home/prior/tmp/groundvla_eval.png")
            # print("Image saved as /home/prior/tmp/groundvla_eval.png")

            # ground vla inference
            act = send_request(img, msg, url)
            print(f"[OpenVLA Client] Received Action (deltas): {act}")

            # step
            next_obs, reward, done, info = env.step(act)
            print(
                f"Time {np.around(time.time()-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)}"
            )
            obs = next_obs
    else:
        raise NotImplementedError

if __name__ == "__main__":
    run_experiment()




