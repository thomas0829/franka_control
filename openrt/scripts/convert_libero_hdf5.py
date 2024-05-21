import datetime
import glob
import os
import pickle
import cv2

import h5py
import hydra
import numpy as np
from tqdm import trange, tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R
import robomimic
from robomimic.utils.lang_utils import get_lang_emb

def quat_to_euler(quat, degrees=False):
        euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
        return euler

def add_angles(delta, source, degrees=False):
        delta_rot = R.from_euler("xyz", delta, degrees=degrees)
        source_rot = R.from_euler("xyz", source, degrees=degrees)
        new_rot = delta_rot * source_rot
        return new_rot.as_euler("xyz", degrees=degrees)

def shortest_angle(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def normalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 2 * (arr - min_val) / (max_val - min_val) - 1


def unnormalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 0.5 * (arr + 1) * (max_val - min_val) + min_val

#just for checking the right keys
def print_hdf5_structure(file_path):
    with h5py.File(file_path, 'r') as file:
        def recurse(group, indent=0):
            for key in group.keys():
                print('  ' * indent + key)
                if isinstance(group[key], h5py.Group):
                    recurse(group[key], indent + 1)
                elif isinstance(group[key], h5py.Dataset):
                    print('  ' * (indent + 1) + "<Dataset>")

        recurse(file)


# Since LIBERO doesn't have language instructions inside the hdf5 file.
def extract_instruction_from_filename(filename):
    instruction = os.path.basename(filename).replace('.hdf5', '')
    replace_patterns = [
        'KITCHEN_SCENE', 'LIVING_ROOM_SCENE', 'STUDY_SCENE', 'demo',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'
    ]
    for pattern in replace_patterns:
        instruction = instruction.replace(pattern, '')
    instruction = instruction.replace('_', ' ')
    instruction = instruction.strip()
    instruction = ' '.join(instruction.split())
    return instruction


@hydra.main(
    config_path="../../configs/", config_name="convert_demos_real", version_base="1.1"
)
def run_experiment(cfg):

    # # REMOVE ME !!!!
    # cfg.data_dir = "/home/marius/Projects/polymetis_franka/data/libero/libero_spatial/"
    # cfg.input_datasets = glob.glob(os.path.join(cfg.data_dir, '*.hdf5'))
    # cfg.output_dataset = "all_libero_spatial"
    # # REMOVE ME !!!!

    # create dataset paths
    dataset_paths = [
        os.path.join(cfg.data_dir, dataset_name) for dataset_name in cfg.input_datasets
    ]
    output_path = os.path.join(cfg.data_dir, cfg.output_dataset)

    print("Processing data ...")

    # create output dataset
    os.makedirs(output_path, exist_ok=True)
    f = h5py.File(os.path.join(output_path, "demos.hdf5"), "w")
    grp_all = f.create_group('data')
    
    # joint input datasets
    episodes = -1
    for i, file_name in tqdm(enumerate(dataset_paths)):
        grp_tmp = h5py.File(file_name, 'r')
        grp = grp_tmp['data']

        # get instruction
        language_instruction = extract_instruction_from_filename(file_name.split("/")[-1])
        # get instruction from embedding
        lang_emb = get_lang_emb(language_instruction)

        if i == 0:
            print("obs keys", grp["demo_0"]["obs"].keys())
            
        for j, name in enumerate(grp):
            episodes += 1
            name_mod = f"demo_{episodes}"
            grp_tmp.copy(f'data/{name}', grp_all, name=name_mod)
            
            # get episode length
            first_key = list(grp_all[name_mod]["obs"].keys())[0]
            episode_len = len(grp_all[name_mod]["obs"][first_key])

            # store language instruction -> still issues w/ string format :(
            # dt = h5py.string_dtype(encoding='utf-8')
            # language_instructions = np.array([language_instruction] * episode_len, dtype=dt)[:,None]
            # grp_all[name_mod]["obs"].create_dataset("language_instruction", language_instructions, dtype=dt)

            # store language embedding from instruction
            lang_embs = np.tile(lang_emb, (episode_len, 1))
            grp_all[name_mod]["obs"].create_dataset("lang_embed", data=lang_embs)
            
        grp_tmp.close()

    # add metadata
    grp_all.attrs["episodes"] = episodes
    grp_all.attrs["env_args"] = '{"env_type":  "blub", "type": "blub"}'
    grp_all.attrs["type"] = "blub"
    
    now = datetime.datetime.now()
    grp_all.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp_all.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)

    # close output dataset
    f.close()

    print("Saved at: {}".format(output_path))

if __name__ == "__main__":
    run_experiment()