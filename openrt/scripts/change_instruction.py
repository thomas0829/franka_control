import datetime
import glob
import os
import pickle
import cv2

import h5py
import hydra
import numpy as np
from tqdm import trange
from robomimic.utils.lang_utils import get_lang_emb

def shortest_angle(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def normalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 2 * (arr - min_val) / (max_val - min_val) - 1


def unnormalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 0.5 * (arr + 1) * (max_val - min_val) + min_val


@hydra.main(
    config_path="../../configs/", config_name="convert_demos_real", version_base="1.1"
)
def run_experiment(cfg):
    from datetime import date
    todays_date = date.today() 
    cfg.data_dir = f"{cfg.data_dir}{todays_date.month}{todays_date.day}/"

    # create dataset paths
    dataset_paths = [
        os.path.join(cfg.data_dir, dataset_name) for dataset_name in cfg.input_datasets
    ]

    list_of_instructions = [
        "pick up the red paprika and put it in the pan",
        "pick up the corn and put it in the pan",
        "put the ketchup in the pot",
        "pick up the yellow paprika and put it in the sink",
        "pick up the cupcake and put it in the pan",
        "pick up the pepper and put it in the pizza",
        "pick up the salmon sushi and put it next to the black block",
        "pick up the corn and put it on the pizza",
        "pick up the grape and put it on the plate",
        "pick up the black cube and put it on the green plate",
        "pick up the salt and put it in the sink",
        "pick up the carrot and put it in the pot",
        "pick up the red paprika and put it on the shelf",
        "put the ketchup next to the paprika",
        "pick up the milk and put it on the burner",
        "pick up the pan",
        "put the mushroom in the pot",
        "put the green cube on the pan",
        "pick up the icecream and put it on the top left burner",
        "pick up the pink duck and put it in the pan",
        "pick up the carrot and put it in the pot",
        "pick up the carrot and put it on the pot",
        "lift the pan and put it in the sink",
        "pick up the red ball and put it in the shelf",
        "pick up the carrot and put it on the right side of the paprika",
        "no instruction",
        "pick up the strawberry and put it next to the ketchup",
        "pick up the green lego block and put it in the sink",
        "stack the small blue cube on top of the bigger blue cube",
        "pick up the small blue cube and put it on top of the green cube",
        "pick up the red ball and put it in the donut",
        "rotate the yellow paprika",
        "pick up the pot from the sink and put it on top of the burner",
        "pick up the strawberry and put it on the plate",
        "pick up the sauce and put it on the plate",
        "pick up the yellow paprika and put it on top of the upside down pot",
        "pick up the cupcake and put it in the sink",
        "flip the pot up right",
        "lift the pan up and put it down",
        "pick up the milk and put it in the pot"
    ]
    for split in cfg.splits:

        for dataset_path in dataset_paths:

            print(f"Loading {dataset_path} {split} ...")

            # gather filenames
            file_names = glob.glob(os.path.join(dataset_path, split, "episode_*.npy"))

            for i in trange(len(file_names)):
                # WARNING: please do NOT add try except -> it is skipping data points and hides the real issues
                # try:

                # load data
                print(file_names[i])
                data = np.load(file_names[i], allow_pickle=True)
                for d in data:
                    d["language_instruction"] = list_of_instructions[i]
                    print(d["language_instruction"])
                #np.save(file_names[i], data, allow_pickle=True)

            


if __name__ == "__main__":
    run_experiment()
