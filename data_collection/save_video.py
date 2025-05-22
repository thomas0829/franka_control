import glob
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
import os
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





