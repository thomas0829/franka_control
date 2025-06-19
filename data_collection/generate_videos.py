


def main():
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str)
    # args = parser.parse_args()
    
    
    # data_dir = args.data_dir
    data_dir = "/home/prior/data_collection/data/date_55"
    npy_dir = f"{data_dir}/npy"
    
    import os
    import imageio
    import glob
    npy_dirs = glob.glob(f"{data_dir}/npy/*/*/episode_*.npy")
    

    task_set = set()
    for npy_path in npy_dirs:
        task_name = "_".join(npy_path.split("/")[-3].split("_")[:-1])
        if task_name == "":
            pass
        
        task_set.add(task_name)

    new_npy_dirs = []
    for task_name in task_set:
        npy_path = f"{data_dir}/npy/{task_name}_2/train/episode_5.npy"
        if os.path.exists(npy_path):
            new_npy_dirs.append(npy_path)
            
    print(f"Found {len(new_npy_dirs)} npy files")
    breakpoint()
    
    import numpy as np
    from tqdm import tqdm
    for npy_path in tqdm(new_npy_dirs):
        ep = np.load(npy_path, allow_pickle=True)
        rgb_sequence = [step["215122253563_rgb"].astype(np.uint8) for step in ep]
        gif_path = npy_path.replace("npy/", "gifs/").replace("train/", "").replace(".npy", ".gif")
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        imageio.mimsave(gif_path, rgb_sequence, fps=15)
        
        print(f"Saved gif to {gif_path}")


if __name__ == "__main__":
    main()
    