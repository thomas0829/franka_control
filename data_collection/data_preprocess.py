
import glob
import re
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import time 
import json
import imageio
from PIL import Image, ImageDraw, ImageFont
import cv2

# CAM_NAME = "215122252864_rgb"
SIDE_CAM_NAME_1 = "213522250587_rgb"
SIDE_CAM_NAME_2 = "215122252864_rgb"
WRIST_CAM_NAME = "332322073412_rgb"

LEFT_CAM = "left_cam_rgb"
RIGHT_CAM = "right_cam_rgb"

def save_img(img, npy_path, img_name, index):
    # Convert the image to a PIL Image object
    img = Image.fromarray(img)
    # 4 digits for index
    img_path = npy_path.replace(".npy", "").replace("npy", img_name).replace("household", "household_json") + f"/{index:04d}.png"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    img.save(img_path)

def action_preprocessing(dic, actions, verbose=False):
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

    if verbose:
        print(f'Action min & max: {actions[...,:6].min(), actions[...,:6].max()}')

    return actions

def shortest_angle(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi
 

def get_episode_data(npy_path: str) -> dict:
    """
    Get the trajectory from the npy file.
    """
    data = np.load(npy_path, allow_pickle=True)

    # stack data
    dic = {}
    obs_keys = data[0].keys()

    for key in obs_keys:
        dic[key] = np.stack([d[key] for d in data])

    actions = np.stack([d["action"] for d in data])
    actions = action_preprocessing(dic, actions)
    episode_data = {
        SIDE_CAM_NAME_1: dic[SIDE_CAM_NAME_1],
        SIDE_CAM_NAME_2: dic[SIDE_CAM_NAME_2],
        WRIST_CAM_NAME: dic[WRIST_CAM_NAME],
        "task": dic["language_instruction"],
        "abs_ee": dic["lowdim_ee"],
        "abs_qpos": dic["lowdim_qpos"],
        "del_action": actions,
    }
    return episode_data

def save_pair(i, side_cam_img_1, side_cam_img_2, npy_path):
    save_img(side_cam_img_1, npy_path, LEFT_CAM, i)
    save_img(side_cam_img_2, npy_path, RIGHT_CAM, i)

def save_rgb_metadata(npy_path: str, verbose=False):
    """
    Save the rgb images and metadata for a given npy file.
    """
    episode_data = get_episode_data(npy_path)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(save_pair, i, img1, img2, npy_path)
            for i, (img1, img2) in enumerate(zip(episode_data[SIDE_CAM_NAME_1], episode_data[SIDE_CAM_NAME_2]))
        ]
        for f in tqdm(futures):
            f.result()  # Wait for all to complete (also raises exceptions if any)
    end_time = time.time()
    if verbose:
        print(f"Time taken: {end_time - start_time} seconds")
    
    # save metadata
    del episode_data[SIDE_CAM_NAME_1]
    del episode_data[SIDE_CAM_NAME_2]
    del episode_data[WRIST_CAM_NAME]

    # save metadata
    metadata_path = npy_path.replace(".npy", "").replace("npy", "metadata").replace("household", "household_json") + ".pkl"
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "wb") as f:
        pickle.dump(episode_data, f)
    if verbose:
        print(f"Saved metadata to {metadata_path}")

def save_json(npy_path: str, verbose=False):
    """
    Save the json file for a given npy file.
    """
    json_data = []
    metadata_path = npy_path.replace(".npy", "").replace("npy", "metadata").replace("household", "household_json") + ".pkl"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    for i in range(len(metadata["del_action"])):
        side_cam_img_1 = npy_path.replace(".npy", "").replace("npy", LEFT_CAM).replace("household", "household_json") + f"/{i:04d}.png"
        side_cam_img_2 = npy_path.replace(".npy", "").replace("npy", RIGHT_CAM).replace("household", "household_json") + f"/{i:04d}.png"
        if not os.path.exists(side_cam_img_1) or not os.path.exists(side_cam_img_2):
            raise FileNotFoundError(f"Image file not found at {side_cam_img_1} or {side_cam_img_2}")
        json_data.append({
            "image_path_0": side_cam_img_1,
            "image_path_1": side_cam_img_2,
            "abs_ee_action": metadata["abs_ee"][i].tolist(),
            "del_ee_action": metadata["del_action"][i].tolist(),
            "task": metadata["task"][i],
        })

    # Save the JSON data to a file
    output_json_path = npy_path.replace(".npy", "").replace("household", "household_json").replace("npy", "json")+".json"

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    if verbose:
        print(f"Data saved to {output_json_path}")

def save_video(npy_path: str, verbose=False):
    """
    Save the video for a given npy file.
    """
    demo = np.load(npy_path, allow_pickle=True)
    language_instruction = demo[0]["language_instruction"]
    if "note" in demo[0].keys():
        note = demo[0]["note"]
    else:
        note = ""

    video_path = npy_path.replace("household", "household_json").replace("npy/", "video/").replace(".npy", f"_{language_instruction.replace(' ', '_')}.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    video_frames = []
    for obs in demo:
        frame = np.concatenate([obs[key] for key in obs if "rgb" in key], axis=1)
        # Convert to PIL image for drawing
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        # Add text
        draw.text((10, 50), f"Instruction: {language_instruction}", fill=(0, 0, 0), font=font)
        draw.text((10, 80), f"Note: {note}", fill=(0, 0, 0), font=font)
        
        # Convert back to np.array
        video_frames.append(np.array(image))
    imageio.mimsave(video_path, video_frames)
    if verbose:
        print(f"save {video_path}")

def process_file(npy_path, file_type):
    if file_type == "1":
        save_video(npy_path)
    elif file_type == "2":
        save_rgb_metadata(npy_path)
    else:
        save_json(npy_path)
      
from natsort import natsorted
  
def main():
    debug = True
    if debug:
        root_dir = "/Volumes/X9 Pro/backup_sample"
        task_name = "pick_fruits_100"
    else:
        root_dir = input("(1) moDataset, (2) X9 Pro: ")
        while root_dir != "1" and root_dir != "2":
            root_dir = input("(1) moDataset, (2) X9 Pro: ")
        if root_dir == "1":
            root_dir = "/Volumes/moDataset/household"
        else:
            # root_dir = "/Volumes/X9 Pro/household"
            root_dir = "/Volumes/X9 Pro/backup_sample"
        task_name = input("task_name: ")
    # demo_npy_paths = glob.glob(f"{root_dir}/{task_name}/npy/*/train/episode_*.npy")
    demo_npy_paths = glob.glob(f"{root_dir}/{task_name}/npy/*.npy")
    # Sort based on the episode number extracted from the file name
    # data_dirs = sorted(demo_npy_paths, key=lambda x: int(re.search(r'episode_(\d+)', x).group(1)))
    data_dirs = natsorted(demo_npy_paths)
    print("First 5 paths:", data_dirs[:5])
    print("Total number of paths:", len(data_dirs))
    
    if debug:
        data_dirs = data_dirs[:10]

    file_type = input("(1) video, (2) image and metadata (3) json ")
    while file_type != "1" and file_type != "2" and file_type != "3":
            file_type = input("(1) video, (2) image and metadata (3) json ")

    # # Parallel processing with tqdm
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(process_file, path, file_type) for path in data_dirs]
    #     for _ in tqdm(as_completed(futures), total=len(futures)):
    #         pass
        
    # log images
    for npy_path in tqdm(data_dirs):    
        if file_type == "1":
            save_video(npy_path)
        elif file_type == "2":
            save_rgb_metadata(npy_path)
        else:
            save_json(npy_path)

if __name__ == "__main__":
    main()