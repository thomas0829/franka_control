

from glob import glob
import numpy as np

demo_filepaths = glob("/home/robots/yuquand/data/date_63_single_img/npy/*/train/episode_*.npy")


for demo_filepath in demo_filepaths:
    episode = np.load(demo_filepath, allow_pickle=True)
    
    # rgb_keys = [key for key in episode.keys() if key.contains("rgb")]

    for observation in episode:
        del observation["215122252864_rgb"]
        del observation["215122252864_depth"]
        del observation["332322073412_rgb"]
        del observation["332322073412_depth"]
    
    # breakpoint()
    np.save(demo_filepath, episode)
    print(f'saved {demo_filepath}')