import glob
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
import os
<<<<<<< HEAD
import argparse
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor

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

    # video_path = npy_path.replace("household", "household_json").replace("npy/", "video/").replace(".npy", f"_{language_instruction.replace(' ', '_')}.mp4")
    video_path = npy_path.replace("npy/", "video/").replace(".npy", f"_{language_instruction.replace(' ', '_')}.mp4")
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
        
def main():
    root_dir = input("(1) moDataset, (2) X9 Pro: ")
    while root_dir != "1" and root_dir != "2":
        root_dir = input("(1) moDataset, (2) X9 Pro: ")
    if root_dir == "1":
        root_dir = "/Volumes/moDataset/household"
    else:
        root_dir = "/Volumes/X9 Pro/household"
    task_name = input("task_name: ")
    demo_npy_paths = glob.glob(f"{root_dir}/{task_name}/npy/*/train/episode_*.npy")
    print(f"found {len(demo_npy_paths)} demos")

    # parallelize
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(save_video, demo_npy_paths), total=len(demo_npy_paths)))
    

if __name__ == "__main__":
    main()
=======


dirs = "/media/prior/moDataset/molmoact_dataset/date_*/npy/*/train/episode_1.npy"
demo_npy_paths = glob.glob(dirs)

filter_words = ["debug", "test", "try_it_out", "practice"]

for episode_path in tqdm(demo_npy_paths):
    if any(word in episode_path for word in filter_words):
        continue
    
    demo = np.load(episode_path, allow_pickle=True)
    if len(demo) == 0:
        continue

    language_instruction = demo[0]["language_instruction"].replace(" ", "_")
    logdir = episode_path.replace("npy/", "").replace("train/", "").replace(".npy", f"_{language_instruction}.mp4").replace("/episode", "_episode")
    logdir = logdir.replace(episode_path.split("/")[-5], "videos")

    os.makedirs(os.path.dirname(logdir), exist_ok=True)
    # breakpoint()
    
    video_frames = []
    for obs in demo:
        video_frame = np.concatenate([obs[key] for key in obs.keys() if "rgb" in key], axis=1)
        video_frames.append(video_frame)
    imageio.mimsave(logdir, video_frames)
    
    print(f"save {logdir}")


# print(f"found {len(demo_npy_paths)} demos")
# for episode_path in tqdm(demo_npy_paths):
#     demo = np.load(episode_path, allow_pickle=True)
#     language_instruction = demo[0]["language_instruction"].replace(" ", "_")
#     # logdir = "/home/robots/yuquand/video/debug.mp4"
#     logdir = episode_path.replace("npy/", "video/").replace("train/", "").replace(".npy", f"_{language_instruction}.mp4").replace("/episode", "_episode")

#     os.makedirs(os.path.dirname(logdir), exist_ok=True)
#     # breakpoint()
    
#     video_frames = []
#     for obs in demo:
#         video_frame = np.concatenate([obs[key] for key in obs.keys() if "rgb" in key], axis=1)
#         video_frames.append(video_frame)
#     imageio.mimsave(logdir, video_frames)
    
#     print(f"save {logdir}")

#     # breakpoint()
>>>>>>> af979c6 ([EDIT] graspmolmo script; cam calibration; video saving; simulation teleop; two langs; new data saving format)





