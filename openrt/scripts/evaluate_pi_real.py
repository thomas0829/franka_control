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

# openpi
from openpi_client import image_tools
from openpi_client import websocket_client_policy


def send_request(images, task_instruction: str, client, state):
    """
    Send the captured image and instruction to the inference server using json_numpy.
    Returns the action output as received from the server.
    """
    front_img = images[0]
    # left_wrist_img = images[1]
    wrist_img = images[1]
    # right_wrist_img = images[2]

    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(front_img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        # "observation/right_wrist_image": image_tools.convert_to_uint8(
        #     image_tools.resize_with_pad(right_wrist_img, 224, 224)
        # ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]
    return action_chunk

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


# def action_preprocessing(dic, actions, interval=1, lowdim_key="lowdim_ee"):
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
#         dic[lowdim_key][interval:, ..., :6] - dic[lowdim_key][:-interval, ..., :6]
#     )

#     # Keep only actions that have valid deltas
#     actions = actions_tmp[:-interval]

#     # Compute shortest angle to avoid wrap-around issues
#     actions[..., 3:6] = shortest_angle(actions[..., 3:6])

#     print(f'Action min & max: {actions[..., :6].min(), actions[..., :6].max()}')

#     return actions

def action_preprocessing(dic, actions, lowdim_key):
        # compute actual deltas s_t+1 - s_t (keep gripper actions)
    actions_tmp = actions.copy()
    actions_tmp[:-1, ..., :6] = (
        dic[lowdim_key][1:, ..., :6] - dic[lowdim_key][:-1, ..., :6]
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

def invert_gripper(chunk):
    curr_gripper = chunk[-1]
    if curr_gripper > 0.5: 
        curr_gripper = 0
    else:
        curr_gripper = 1
    
    result = chunk.copy()
    result[-1] = curr_gripper
    return result

def replay_episode(demo, env, length=-1, visual=False):
    # stack data
    # dic = get_dict(demo)
    dic = demo
    actions = np.stack(demo["action"])
    # actions = action_preprocessing(dic, actions, interval=interval) # delta action

    left_actions = action_preprocessing(dic, actions[:, :7], lowdim_key="l_lowdim_ee") # delta action
    right_actions = action_preprocessing(dic, actions[:, 7:], lowdim_key="r_lowdim_ee") # delta action

    demo_length = actions.shape[0]

    if length == -1:
        length = demo_length
    else:
        length = length

    for step_idx in tqdm(range(0, length-1)):
        # act = actions[step_idx]
        obs = env.get_observation()

        if visual:
            cv2.imshow("Camera View", cv2.cvtColor(obs["213522250587_rgb"], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)  # Small delay to allow the image to refresh
        
        print(f" Action Step: {step_idx} | Left Action: {left_actions[step_idx][-1]} | Right Action: {right_actions[step_idx][-1]}")
        # next_obs, reward, done, info = env.step(act)
        env.step_all(left_actions[step_idx], right_actions[step_idx])

        # print("action: ", act)
    cv2.destroyAllWindows()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)

# config_name="collect_demos_real" for real robot config_name="collect_demos_sim" for simulation
@hydra.main(
    # config_path="../../configs/", config_name="collect_demos_real", version_base="1.1"
    # config_path="../../configs/", config_name="eval_openvla_real", version_base="1.1"
    # config_path="../../configs/", config_name="eval_molmoact_real_bimanual", version_base="1.1"
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
    # env = make_env(
    #     robot_cfg_dict=hydra_to_dict(cfg.robot),
    #     env_cfg_dict=hydra_to_dict(cfg.env) if "env" in cfg.keys() else None,
    #     seed=cfg.seed,
    #     device_id=0,
    #     verbose=True,
    # )


    # intializer pi client
    # client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

    # url = "dfcb5c292024.ngrok-free.app"
    # url = cfg.url
    # url = "9ddb8363701b.ngrok-free.app"
    url = cfg.url
    client = websocket_client_policy.WebsocketClientPolicy(
        host=url,
        port=443,
    )

    # create env
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
        # bimanual=cfg.bimanual,
    )

    camera_names = [k for k in env.get_images().keys()]

    print(f"Camera names: {camera_names}")

    # 213522250587
    # 218622276072
    # 128422272697

    # TODO: (yuquan) implment cropping!
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
    #     return image
    
    env.seed(cfg.seed)
    # prev_gripper_left = 0
    # prev_gripper_right = 0

    prev_gripper = 0


    

    # mode = "close_loop" # replay, close_loop, open_loop
    # mode = "close_loop" # replay, close_loop, open_loop
    mode = input("Please enter the mode (replay, close_loop, open_loop, mix): ").strip().lower()
    
    # demo_dir = "/home/prior/data_collection/data/38/pick_duster_npy/pick_duster_1/train/episode_2.npy"
    # demo_dir = "/home/prior/rlds_dataset_builder/pick_mango/data/pick_mango_1/train/episode_2.npy"
    # demo_dir = "/home/prior/yuquand/data/79/lift_tray_exp_0/train_pickle/000002.pkl"
    # demo_train_dir = "/home/prior/yuquand/data/79/lift_tray_exp_0/train"

    # demo_train_dir = "/home/prior/yuquand/data/711_franka/lift_tray_exp_0/train"
    demo_train_dir = "/home/prior/yuquand/data/711/Set_table_exp_0/train"
    demo_dir = f"{demo_train_dir}_pickle/000007.pkl"

    # camera_name = "213522250587"
    # camera_names = ["213522250587", "218622276072", "128422272697"]

    CAMERA2NAMES = {
        "wrist": "128422272697_rgb",
        "front": "215122252864_rgb"
    }

    camera_name = "215122252864"
    camera_names = ["215122252864", "128422272697"]
    print("[INFO] mode: ", mode)

    print("[INFO] msg: ", cfg.msg)
    print("[INFO] camera_names: ", camera_names)
    if mode == "replay":
        while True:
            try:
                obs = env.reset()
                reset = input("Waiting for reset... Press Enter to continue.")

                print('[INFO] demo_dir: ', demo_dir)
                demo = np.load(demo_dir, allow_pickle=True)
                print(demo.keys())
                replay_episode(demo, env, visual=True)
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Exiting replay mode.")
                obs = env.reset()
                # breakpoint() 
    elif mode == "close_loop":
        while True:
            obs = env.reset()

            task_name = input("Please enter the task name: ")
            episode = input("Please enter the episode number: ")
            # path_name = f"/home/prior/yuquand/outputs/molmoact_bi_eval/pi/{task_name}"
            path_name = f"/home/robots/yuquand/outputs/molmoact_eval/pi/{task_name}"
            os.makedirs(path_name, exist_ok=True)

            reset = input("Waiting for reset... Press Enter to continue.")
            # path_name = input("Please enter the path to the demo file: ")

            saved_dic = {}
            try:
                for i in range(cfg.traj_length): 
                    print(f'Step: {i} | lang: {cfg.msg}')
                    # img = image_preprocessing(obs[camera_names[0]+"_rgb"])
                    imgs = []

                    for camera_name in camera_names:
                        img = obs[camera_name+"_rgb"]
                        # img = image_preprocessing(img)
                        imgs.append(img)

                    msg = cfg.msg
                    print(f"Sending message: {msg}")
                    # sanity check            
                    from PIL import Image
                    Image.fromarray(imgs[0]).save("/home/robots/tmp/scene_camera.png")
                    Image.fromarray(imgs[1]).save("/home/robots/tmp/wrist_camera.png")
                    print("Image saved as /home/robots/tmp/scene_camera.png")
                    print("Image saved as /home/robots/tmp/wrist_camera.png")

                    start = time.time() 
                    # state = np.concatenate([obs["l_lowdim_qpos"], obs["r_lowdim_qpos"]], axis=0)
                    state = obs["lowdim_qpos"]
                    act = send_request(imgs, msg, client, state=state)
                    
                    end = time.time()

                    # agent.act()
                    for i in range(len(act)):
                        chunk = act[i]
                        # left_action, right_action = chunk[:7], chunk[7:]
                        # left_action = invert_gripper(left_action)
                        # right_action = invert_gripper(right_action)

                        # step
                        # next_obs, reward, done, info = env.step_all(left_action, right_action, verbose=True)
                        # next_obs, reward, done, info = env.step(chunk)

                        # if prev_gripper_left != left_action[-1] or prev_gripper_right != right_action[-1]:
                            # time.sleep(2)


                        chunk = act[i]
                        # chunk = invert_gripper(chunk)
                        next_obs = env.step(chunk)[0]
                        if prev_gripper != chunk[-1]:
                            time.sleep(2)
                        obs = next_obs 
                        prev_gripper = chunk[-1]
                    
                        print(
                            f"Time {np.around(end-start, 3)/ len(act)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(chunk,3)}"
                        )
                        for k, v in obs.items():
                            if "depth" in k:
                                continue
                            if k not in saved_dic:
                                saved_dic[k] = []
                            saved_dic[k].append(v)
                        # prev_gripper_left, prev_gripper_right = left_action[-1], right_action[-1]
                    
                        # print(
                        #     f"Time {np.around(end-start, 3)/ len(act)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(chunk,3)}"
                        # )

            except KeyboardInterrupt:
                print("Saving episode...")
                save_path = os.path.join(path_name, f"{episode}.pkl")

                with open(save_path, "wb") as f:
                    pickle.dump(saved_dic, f)
                    print(f"[INFO] Demo saved to {save_path}")
                
                del client
                client = websocket_client_policy.WebsocketClientPolicy(
                    host=url,
                    port=443,
                )
                print("[INFO] Client reinitialized.")
                continue


    elif mode == "open_loop":
        obs = env.reset()
        reset = input("Waiting for reset... Press Enter to continue.")
        print('[INFO] demo_dir: ', demo_dir)
        demo = np.load(demo_dir, allow_pickle=True)
        episode_idx = demo_dir.split("/")[-1].split(".")[0]
        camera_paths = {}
        for camera_name in camera_names:
            camera_paths[camera_name] = os.path.join(demo_train_dir, camera_name+"_rgb", episode_idx)
        # breakpoint()
        inference_mean = []
        for i in range(0, len(demo["action"]), 8):
            print(f'Step: {i} | lang: {cfg.msg}')
            start = time.time()
            imgs = []

            image_order = ["213522250587", "218622276072", "128422272697"]
            for camera_name in image_order:
                raw_img = cv2.imread(camera_paths[camera_name] + f"/{i:06d}.png")
                img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                # img = image_preprocessing(img)
                # save image for debugging
                debug_save_path = f"/home/prior/tmp/molmoact_eval"
                os.makedirs(os.path.join(debug_save_path, camera_name), exist_ok=True)
                Image.fromarray(img).save(os.path.join(debug_save_path, camera_name, f"{i:06d}.png"))
                imgs.append(img)

            # breakpoint()
            # msg = cfg.msg
            msg = cfg.msg
            print(f"Sending message: {msg}")
            
            raise NotImplementedError
            act = send_request(imgs, msg, url)

            # action chunking execute
            # for i in range(len(act)):
            #     chunk = act[i]
            #     left_action, right_action = chunk[:7], chunk[7:]
            #     left_action = invert_gripper(left_action)
            #     right_action = invert_gripper(right_action)
            #     # step
            #     _, _, _, _ = env.step_all(left_action, right_action)
            #     if prev_gripper_left != left_action[-1] or prev_gripper_right != right_action[-1]:
            #         time.sleep(2)
            #     prev_gripper_left, prev_gripper_right = left_action[-1], right_action[-1]
            
            for i in range(len(act)):
                chunk = act[i]
                left_action, right_action = chunk[:7], chunk[7:]
                left_action = invert_gripper(left_action)
                right_action = invert_gripper(right_action)

                print("Action: ", left_action, right_action)

                # step
                next_obs, reward, done, info = env.step_all(left_action, right_action, verbose=True)
                if prev_gripper_left != left_action[-1] or prev_gripper_right != right_action[-1]:
                    time.sleep(2)
                prev_gripper_left, prev_gripper_right = left_action[-1], right_action[-1]

            inference_mean.append(act)
            print(
                f"Time {np.around(time.time()-start, 3)} | Mean: {np.array(inference_mean).mean()} | Act {act}"
            )
    else:
        raise NotImplementedError

if __name__ == "__main__":
    run_experiment()



