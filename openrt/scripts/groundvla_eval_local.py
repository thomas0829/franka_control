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
from typing import List

# GroundVLA imports
from PIL import Image
import requests
import json_numpy
json_numpy.patch()
import re

# camera_id = "213522250587_rgb"
# CAMERA2NAMES = {
#     "side": "215122256044_rgb", # side view
#     "front": "213522250587_rgb", # front view
#     "wrist": "128422272697_rgb", # wrist view
# }

CAMERA2NAMES = {
    "wrist": "128422272697_rgb",
    "front": "215122252864_rgb"
}


from concurrent.futures import ThreadPoolExecutor, as_completed
executor = ThreadPoolExecutor(max_workers=1)

def send_request(images: List[np.ndarray], instruction: str, server_url: str, multi_views: bool = False):
    """
    Send the captured image and instruction to the inference server using json_numpy.
    Returns the action output as received from the server.
    """
    if not multi_views:
        # Convert PIL image to a NumPy array
        image_np = np.array(images[0])
        
        # Prepare the payload with the image and instruction from the script
        payload = {
            "image": image_np, # scene cam
            "instruction": instruction
        }
    else:
        # Convert PIL image to a NumPy array
        scene_img_np = np.array(images[0])
        wrist_img_np = np.array(images[1])

        # Image.fromarray(scene_img_np).save("/home/prior/tmp/scene_cam.png")
        Image.fromarray(scene_img_np).save("/home/robots/tmp/scene_cam.png")
        # Image.fromarray(wrist_img_np).save("/home/prior/tmp/wrist_cam.png")
        Image.fromarray(wrist_img_np).save("/home/robots/tmp/wrist_cam.png")
        
            
        # Prepare the payload with the image and instruction from the script
        payload = {
            "image1": scene_img_np, # scene cam
            "image2": wrist_img_np, # wrist cam
            "timestamp": time.time(), # add timestamp for debugging
            "instruction": instruction
        }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(server_url, headers=headers, data=json_numpy.dumps(payload))

    if response.status_code != 200:
        raise Exception(f"Server error: {response.text}")
    
    print("response: ", response)
    response_data = response.json()

    # import pdb;pdb.set_trace()
    # action = json_numpy.loads(response_data)
    # print(action)
    return response_data

    # headers = {"Content-Type": "application/json"}

    # while True:
    #     response = requests.post(
    #         server_url,
    #         headers=headers,
    #         data=json_numpy.dumps(payload),
    #         timeout=120,          # keep whatever timeout you prefer
    #     )

    #     if response.status_code != 200:
    #         raise Exception(f"Server error: {response.text}")

    #     try:
    #         response_data = response.json()
    #     except ValueError:
    #         # server sent non-JSON (e.g. empty) – try again
    #         time.sleep(2)
    #         continue

    #     # The lightweight endpoints often reply {"status": "..."}
    #     # while the heavy inference is still running.  Break out and
    #     # return only once we get something OTHER than the status ping.
    #     print(response_data)
    #     if "status" not in response_data:
    #         break

    #     # still a status-only ping → wait a moment and poll again
    #     time.sleep(2)

    # return response_data





# def send_request(image: Image.Image, instruction: str, server_url: str):
#     """
#     Send the captured image and instruction to the inference server using json_numpy.
#     Returns the action output as received from the server.
#     """
#     # Convert PIL image to a NumPy array
#     image_np = np.array(image)
#     # Prepare the payload with the image (as 'full_image') and instruction
#     payload = {
#         "image": {
#              "full_image": image_np
#         },
#         "instruction": instruction
#     }
#     headers = {"Content-Type": "application/json"}
#     response = requests.post(server_url, headers=headers, data=json_numpy.dumps(payload))
#     if response.status_code != 200:
#         raise Exception(f"Server error: {response.text}")
#     print("response: ", response)
#     response_data = np.array(response.json())
#     print("response_data: ", response_data)
    
#     return response_data

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

def action_preprocessing(dic, actions):
        # compute actual deltas s_t+1 - s_t (keep gripper actions)
    actions_tmp = actions.copy()
    actions_tmp[:-1, ..., :6] = (
        dic["lowdim_ee"][1:, ..., :6] - dic["lowdim_ee"][:-1, ..., :6]
    )
    actions = actions_tmp[:-1]
    

        # compute shortest angle -> avoid wrap around
    actions[..., 3:6] = shortest_angle(actions[..., 3:6])

    # real data source
    #actions[..., [3,4,5]] = actions[..., [4,3,5]]
    #actions[...,4] = -actions[...,4]
    # actions[...,3] = -actions[...,3] this is a bug

    print(f'Action min & max: {actions[...,:6].min(), actions[...,:6].max()}')

    return actions

# def action_preprocessing(dic, actions, interval=1):
#     """
#     Compute action deltas based on a specified interval.
    
#     Args:
#         dic (dict): Dictionary containing stacked episode data.
#         actions (np.array): Original actions of shape (N, 7).
#         interval (int): Step size for computing differences (default=1).
    
#     Returns:
#         np.array: Processed actions of shape (N - interval, 7).
#     """
#     actions_tmp = actions.copy()

#     # Compute deltas with interval steps
#     actions_tmp[:-interval, ..., :6] = (
#         dic["lowdim_ee"][interval:, ..., :6] - dic["lowdim_ee"][:-interval, ..., :6]
#     )

#     # Keep only actions that have valid deltas
#     actions = actions_tmp[:-interval]

#     # Compute shortest angle to avoid wrap-around issues
#     actions[..., 3:6] = shortest_angle(actions[..., 3:6])

#     print(f'Action min & max: {actions[..., :6].min(), actions[..., :6].max()}')

#     return actions


    
def replay_episode(demo, env, visual=False):
    # stack data
    dic = get_dict(demo)
    actions = np.stack([d["action"] for d in demo])
    actions = action_preprocessing(dic, actions) # delta action
    demo_length = actions.shape[0]

    for step_idx in tqdm(range(demo_length)):
        act = actions[step_idx]

        obs = env.get_observation()

        if visual:
            cv2.imshow("Camera View", obs["215122252864_rgb"])
            cv2.waitKey(1)  # Small delay to allow the image to refresh
        
        if step_idx == 15:
            breakpoint() 
        env.step(act)
    cv2.destroyAllWindows()

def replay_episode_pickle(demo_dir, env, visual=False):
    # stack data
    # dic = get_dict(demo)
    with open(demo_dir, 'rb') as f:
        demo = pickle.load(f)
    
    # actions = np.stack([d["action"] for d in demo])
    actions = demo["action"]
    actions = action_preprocessing(demo, actions) # delta action
    demo_length = actions.shape[0]

    for step_idx in tqdm(range(demo_length)):
        act = actions[step_idx]

        obs = env.get_observation()

        if visual:
            cv2.imshow("Camera View", obs["215122252864_rgb"])
            cv2.waitKey(1)  # Small delay to allow the image to refresh
        
        if step_idx == 15:
            breakpoint() 
        env.step(act)
    cv2.destroyAllWindows()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)

def invert_gripper(chunk):
    curr_gripper = chunk[-1]
    print(curr_gripper)
    if curr_gripper > 0.5: 
        curr_gripper = 0
    else:
        curr_gripper = 1
    
    result = chunk.copy()
    result[-1] = curr_gripper
    return result

def model_inference(obs, msg, url, multi_views, camera_id):
    # sanity check            
    # print(obs.keys())
    img = obs[camera_id]
    wrist_img = obs[CAMERA2NAMES["wrist"]]
    act = send_request([img, wrist_img], msg, url, multi_views=multi_views)
    print(f"[Molmo Act Client] Received Action (deltas): {act}")
    return act

import logging
from utils.pointclouds import *
# Execute interpolated motion   
def execute_interpolated_motion(env, curr_pose_matrix, target_pose_matrix, obs, distance_threshold=0.01, verbose=False, visualize=False, sleep_time=0.5, close_gripper_state=False):
    """Interpolates poses and executes them sequentially in the environment."""

    # execute pregrasp
    interpolated_motions = interpolate_6d_poses(curr_pose_matrix, target_pose_matrix, distance_threshold=distance_threshold)
    
    
    # prev_gripper = obs["lowdim_ee"][-1] # WARNING: keep the gripper state same
    # if close_gripper:
    count = 0
    for motion_6d in interpolated_motions[1:]:
        sub_pose = np.zeros(7)
        sub_pose[:6] = motion_6d
        
        if close_gripper_state:
            sub_pose[6] = 1
        else:
            sub_pose[6] = 0 # WARNING: keep the gripper state same


        next_obs, _, _, _ = env.step_with_pose(sub_pose)
        if count == 0:
            time.sleep(3.0)
        prev_gripper = next_obs["lowdim_ee"][-1]
        
        pose_error = np.linalg.norm(next_obs["lowdim_ee"][:6] - sub_pose[:6])
        position_error = np.linalg.norm(next_obs["lowdim_ee"][:3] - sub_pose[:3])
        
        count += 1
        if verbose:
            logging.info(f"pose error: {pose_error}")
            logging.info(f"position error: {position_error}")
            logging.info(f"------------------------------------------------------------------------------------")
    
    time.sleep(sleep_time)
    return next_obs, interpolated_motions

def close_gripper(env, lowdim_ee, time_sleep=3):
    for _ in range(5):
        sub_pose = np.zeros(7)
        sub_pose[:6] = lowdim_ee[:6]
        sub_pose[6] = 1
        env.step_with_pose(sub_pose)
    time.sleep(time_sleep)

# config_name="collect_demos_real" for real robot config_name="collect_demos_sim" for simulation
@hydra.main(
    # config_path="../../configs/", config_name="collect_demos_real", version_base="1.1"
    config_path="../../configs/", config_name="eval_openvla_real_thinkpad", version_base="1.1"
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
    
    # breakpoint()

    # breakpoint()
    # pick the camera_name
    # https://fb25-71-41-244-70.ngrok-free.app


    # # crop image observations
    # if cfg.aug.camera_crop is not None:
    #     env = CropImageWrapper(
    #         env,
    #         x_min=cfg.aug.camera_crop[0],
    #         x_max=cfg.aug.camera_crop[1],
    #         y_min=cfg.aug.camera_crop[2],
    #         y_max=cfg.aug.camera_crop[3],
    #         image_keys=[cn + "_rgb" for cn in camera_names],
    #         crop_render=True,
    #     )

    # resize image observations
    '''
    if cfg.aug.camera_resize is not None:
        env = ResizeImageWrapper(
            env,
            size=cfg.aug.camera_resize,
            image_keys=[cn for cn in camera_names],
        )
    '''

    # def image_preprocessing(image):
    #     image = cv2.resize(image, [360, 360])
        # return image
    
    env.seed(cfg.seed)
    obs = env.reset()
    home_pose = obs["lowdim_ee"]

    mode = "demo" # replay, close_loop, open_loop, demo
    # demo_dir = "/home/prior/dataset/date_618/npy/pick_banana_0/train/episode_15.npy"
    # demo_dir = "/home/prior/dataset/date_618/npy/push_apple_0/train/000000.pkl"
    # demo_dir = "/home/prior/dataset/date_620/push_button_pickle/000000.pkl"
    # root_dir = "/home/prior/dataset/date_620/push_button"
    # episode_number = "000009"
    # root_dir = "/home/prior/dataset/date_620/stack_bowl"
    # episode_number = "000035"
    # root_dir = "/home/prior/dataset/date_621/put_sponge_pink_bowl"
    # episode_number = "000008"
    # root_dir = "/home/prior/dataset/date_621/put_sponge_pink_bowl"
    # episode_number = "000000"
    # root_dir = "/home/robots/dataset/date_623/throw_can"
    # episode_number = "000007"
    # root_dir = "/media/robots/moDataset/date_624/throw_bowl"
    # episode_number = "000029"
    # root_dir = "/media/robots/moDataset/tabletop_date_626/throw_can"
    # episode_number = "000001"
    root_dir = "/media/robots/X9 Pro/date_716/wipe_stain/wipe_stain_0"
    episode_number = "000020"

    
    # root_dir = "/home/prior/dataset/date_621/push_button_failure_recovery"
    # episode_number = "000004"
    demo_dir = f"{root_dir}_pickle/{episode_number}.pkl"
    
    print("[WARN] hardcode demo directory")

    url = cfg.url
    print("[INFO] mode: ", mode)
    print("[INFO] url: ", url)
    print("[INFO] action chunking: ", cfg.action_chunking)
    print("[INFO] msg: ", cfg.msg)
    

    prev_gripper = 0
    # camera_id = "215122256044_rgb" # side view
    if cfg.camera_name not in CAMERA2NAMES.keys():
        raise ValueError(f"Invalid camera name: {cfg.camera_name}. Choose from {list(CAMERA2NAMES.keys())}.")
    camera_id = CAMERA2NAMES[cfg.camera_name]
    print(f"[INFO] Using camera: {camera_id} | Corresponding name: {cfg.camera_name}")
    print(f"multi views: {cfg.multi_views}")
    
    
    # breakpoint()
    if mode == "replay":
        print('[INFO] demo_dir: ', demo_dir)
        # demo = np.load(demo_dir, allow_pickle=True)
        # replay_episode(demo, env, visual=False)
        replay_episode_pickle(demo_dir, env, visual=False)
    elif mode == "close_loop":
        raise NotImplementedError
        
        for i in range(cfg.traj_length): 
            try:
                print(f'Step: {i} | lang: {cfg.msg}')
                start = time.time()
                msg = cfg.msg

                start = time.time()
                try:
                    act = model_inference(obs, msg, url, cfg.multi_views, camera_id)
                except:
                    breakpoint()
                end = time.time()
                if cfg.action_chunking:
                    # execute h actions at once
                    
                    for i in range(len(act)):
                        # chunk = action_queue.pop(0)
                        chunk = act[i]
                        chunk = invert_gripper(chunk)
                        next_obs = env.step(chunk)[0]
                        if prev_gripper != chunk[-1]:
                            time.sleep(2)
                        obs = next_obs 
                        prev_gripper = chunk[-1]
                    
                        print(
                            f"Time {np.around(end-start, 3)/ len(act)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(chunk,3)}"
                        )
                else:       
                    # step
                    next_obs = env.step(act)[0]
                    print(
                        f"Time {np.around(end-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(act,3)}"
                    )
                    obs = next_obs
            except KeyboardInterrupt:
                pre_home_pose_matrix = get_pose_matrix(home_pose).copy()
                pre_home_pose_matrix[:3, 3] -= pre_home_pose_matrix[:3, 2] * 0.12
                close_gripper(env, next_obs["lowdim_ee"], time_sleep=2)
                next_obs, interpolated_motions = execute_interpolated_motion(env, get_pose_matrix(env.get_observation()["lowdim_ee"]), pre_home_pose_matrix, next_obs, verbose=True, sleep_time=2)
                next_obs, interpolated_motions = execute_interpolated_motion(env, get_pose_matrix(env.get_observation()["lowdim_ee"]), get_pose_matrix(home_pose), next_obs, verbose=True, sleep_time=1.5)
                break

    elif mode == "open_loop":
        # front_view_dir = f"/home/prior/dataset/date_620/push_button/{CAMERA2NAMES['front']}/000000"
        # wrist_view_dir = f"/home/prior/dataset/date_620/push_button/{CAMERA2NAMES['wrist']}/000000"
        front_view_dir = f"{root_dir}/{CAMERA2NAMES['front']}/{episode_number}"
        wrist_view_dir = f"{root_dir}/{CAMERA2NAMES['wrist']}/{episode_number}"
        
        
        front_rgb_paths = sorted(glob.glob(os.path.join(front_view_dir, "*.png")))
        wrist_rgb_paths = sorted(glob.glob(os.path.join(wrist_view_dir, "*.png")))
        
        assert len(front_rgb_paths) == len(wrist_rgb_paths), "Front and wrist view images must have the same number of frames."


        with open(demo_dir, 'rb') as f:
            demo = pickle.load(f)
        # breakpoint()
        msg = demo["language_instruction"][0]
        
        for i in range(0, len(front_rgb_paths)-1, 8):

            print(f'Step: {i} | lang: {msg}')
            # msg = cfg.msg

            start = time.time()
            
            training_obs = {}
            training_obs[CAMERA2NAMES["front"]] = Image.open(front_rgb_paths[i])
            training_obs[CAMERA2NAMES["wrist"]] = Image.open(wrist_rgb_paths[i])
            
            
            act = model_inference(training_obs, msg, url, cfg.multi_views, camera_id)
            end = time.time()
            if cfg.action_chunking:
                # execute h actions at once
                # action_queue.extend([chunk for chunk in act])   
                
                for i in range(len(act)):
                    # chunk = action_queue.pop(0)
                    chunk = act[i]
                    chunk = invert_gripper(chunk)
                    next_obs = env.step(chunk)[0]
                    if prev_gripper != chunk[-1]:
                        time.sleep(2)
                    prev_gripper = chunk[-1]
                    print(
                        f"Time {np.around(end-start, 3)/ len(act)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(chunk,3)}"
                    )
    elif mode == "demo":
        while True:
            num_steps = 0
            msg = input("Enter the instruction: ")
            while True: 
                try:
                    print(f'Step: {num_steps} | lang: {msg}')
                    start = time.time()
                    
                    
                    start = time.time()
                    
                    
                    act = model_inference(env.get_observation(), msg, url, cfg.multi_views, camera_id)
                    end = time.time()
                    if cfg.action_chunking:
                        # execute h actions at once
                        
                        for i in range(len(act)):
                            # chunk = action_queue.pop(0)
                            chunk = act[i]
                            chunk = invert_gripper(chunk)
                            next_obs = env.step(chunk)[0]
                            if prev_gripper != chunk[-1]:
                                time.sleep(2)
                            obs = next_obs 
                            prev_gripper = chunk[-1]
                        
                            print(
                                f"Time {np.around(end-start, 3)/ len(act)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(chunk,3)}"
                            )
                    else:       
                        # step
                        next_obs = env.step(act)[0]
                        print(
                            f"Time {np.around(end-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(act,3)}"
                        )
                        obs = next_obs
                    num_steps += 1
                except KeyboardInterrupt:
                    pre_home_pose_matrix = get_pose_matrix(home_pose).copy()
                    pre_home_pose_matrix[:3, 3] -= pre_home_pose_matrix[:3, 2] * 0.12
                    obs, interpolated_motions = execute_interpolated_motion(env, get_pose_matrix(env.get_observation()["lowdim_ee"]), pre_home_pose_matrix, obs, verbose=True, sleep_time=1.5)
                    obs, interpolated_motions = execute_interpolated_motion(env, get_pose_matrix(env.get_observation()["lowdim_ee"]), get_pose_matrix(home_pose), obs, verbose=True, sleep_time=2)
                    env.reset()
                    break
                    



if __name__ == "__main__":
    run_experiment()




