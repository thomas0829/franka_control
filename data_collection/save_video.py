import glob
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
import os
import argparse

# args = 
root_dir = "/home/robots/yuquand/data"
task_name = "cover_lid_food_container"

demo_npy_paths = glob.glob(f"{root_dir}/{task_name}/npy/*/train/episode_*.npy")

print(f"found {len(demo_npy_paths)} demos")


for episode_path in tqdm(demo_npy_paths):
    demo = np.load(episode_path, allow_pickle=True)
    language_instruction = demo[0]["language_instruction"].replace(" ", "_")
    logdir = episode_path.replace("npy/", "video/").replace("train/", "").replace(".npy", f"_{language_instruction}.mp4").replace("/episode", "_episode")
    os.makedirs(os.path.dirname(logdir), exist_ok=True)
    video_frames = []
    for obs in demo:
        video_frame = np.concatenate([obs[key] for key in obs.keys() if "rgb" in key], axis=1)
        video_frames.append(video_frame)
    imageio.mimsave(logdir, video_frames)
    print(f"save {logdir}")





