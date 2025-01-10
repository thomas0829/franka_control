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
    # return 0


def unnormalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 0.5 * (arr + 1) * (max_val - min_val) + min_val


@hydra.main(
    config_path="../../configs/", config_name="convert_demos_real", version_base="1.1"
)
def run_experiment(cfg):
    from datetime import date
    todays_date = date.today() 
    #cfg.data_dir = f"{cfg.data_dir}{todays_date.month}{todays_date.day}/"
    cfg.data_dir = f"{cfg.data_dir}817_robot/"

    # create dataset paths
    dataset_paths = [
        os.path.join(cfg.data_dir, dataset_name) for dataset_name in cfg.input_datasets
    ]

    print("Processing data ...")
    hdf5_path = os.path.join(cfg.data_dir, cfg.output_dataset)
    os.makedirs(hdf5_path, exist_ok=True)
 
    f = h5py.File(os.path.join(hdf5_path, "demos.hdf5"), "w")

    # create data group
    grp = f.create_group("data")
    grp_mask = f.create_group("mask")

    episodes = 0

    demo_keys = {}
  

    for split in cfg.splits:

        demo_keys[split] = []

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

                # stack data
                dic = {}
                try:
                    obs_keys = data[0].keys()
                except:
                    print(f"Error loading {file_names[i]}")
                    continue
                for key in obs_keys:
                    dic[key] = np.stack([d[key] for d in data])
                    
                    # only keep every 3rd timestep
                    if cfg.data_source == "mujoco":
                        dic[key] = dic[key][::2]

                actions = np.stack([d["action"] for d in data])
                
                # only keep every 3rd timestep
                if cfg.data_source == "mujoco":
                    actions = actions[::2]

                if cfg.blocking_control:
                    # compute actual deltas s_t+1 - s_t (keep gripper actions)
                    actions_tmp = actions.copy()
                    actions_tmp[:-1, ..., :6] = (
                        dic["lowdim_ee"][1:, ..., :6] - dic["lowdim_ee"][:-1, ..., :6]
                    )
                    actions = actions_tmp[:-1]

                    # remove last state s_T
                    for key in obs_keys:
                        dic[key] = dic[key][:-1]
                    
                    # remove grasp actions if gripper is not closed -> blocking grasp
                    # first_grasp_idx = np.where(actions[...,-1] == 1)[0][0]
                    # actions[first_grasp_idx:first_grasp_idx+3,-1] = 0

                # create demo group
                demo_key = f"demo_{episodes}"
                demo_keys[split].append(demo_key)
                ep_data_grp = grp.create_group(demo_key)

                # compute shortest angle -> avoid wrap around
                actions[..., 3:6] = shortest_angle(actions[..., 3:6])
                
                #if cfg.data_source == "real":
                    #actions[..., [3,4,5]] = actions[..., [4,3,5]]
                    #actions[...,4] = -actions[...,4]
                    #actions[...,3] = -actions[...,3]

                print(actions[...,:6].min(), actions[...,:6].max())
                print(actions.shape)
                coarse_map = {}
                accum_action = None
                accum_cnt = 0
                for j in range(len(actions)):
                    action = actions[j]
                    if accum_action is None:
                        accum_action = action 
                    else:
                        accum_action += action 
                    if j!=0 and j%15 == 0:
                        coarse_map[accum_cnt] = accum_action
                        accum_action = None
                        accum_cnt += 1
                if accum_action is None:
                    coarse_map[accum_cnt] = coarse_map[accum_cnt-1]
                else:
                    coarse_map[accum_cnt] = accum_action # the last remaining action accums
                actions_accum = []
                for j in range(len(actions)):
                    actions_accum.append(coarse_map[j//15].tolist())
                actions_accum = np.array(actions_accum)
                # add action dataset
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("actions_accum", data=actions_accum)

                # add done dataset
                dones = np.zeros(len(actions)).astype(bool)
                dones[-1] = True
                ep_data_grp.create_dataset("dones", data=dones)

                # create obs and next_obs groups
                ep_obs_grp = ep_data_grp.create_group("obs")
                
                if "language_instruction" not in obs_keys:
                    dic["language_instruction"] = ["pick up the red block"]
                    print("WARNING: 'language_instruction' not in dataset, addind template instruction!")
                obs_keys = dic.keys()

                # print("WARNING: NOT skipping corner camera!")
                # add obs and next_obs datasets
                for obs_key in obs_keys:
                    #if cfg.data_source == "real":
                    #   if obs_key== '215122255213_rgb':
                    #       print("WARNING: skipping 215122255213_rgb!")
                    #       continue # skipping the corner camera
                    obs = dic[obs_key]
                    if "_rgb" in obs_key:
                        # crop images for training
                        # x_min, x_max, y_min, y_max = cfg.aug.camera_crop
                        # obs = obs[:, x_min : x_max, y_min : y_max]
                        # # resize images for training
                        # obs = np.stack([cv2.resize(img, cfg.aug.camera_resize) for img in obs])
                        # print(f"WARNING: replacing '{obs_key}' with 'front_rgb'!")
                        obs_key = "front_rgb"
                    if obs_key == "language_instruction":
                        lang_emb = get_lang_emb(obs[0])
                        lang_emb = np.tile(lang_emb, (len(obs), 1))
                        ep_obs_grp.create_dataset("lang_embed", data=lang_emb)
                        obs = np.array(obs, dtype='S80')

                    ep_obs_grp.create_dataset(obs_key, data=obs)

                ep_data_grp.attrs["num_samples"] = len(actions)
                episodes += 1
                
                # WARNING: please do NOT add try except -> it is skipping data points and hides the real issues
                # except Exception as e:
                #     print(e)
                #     print(f"Error loading {file_names[i]}")

        # create mask dataset
        grp_mask.create_dataset(split, data=np.array(demo_keys[split], dtype="S"))


    # dummy metadata so robomimic is happy
    grp.attrs["episodes"] = episodes
    grp.attrs["env_args"] = '{"env_type":  "blub", "type": "blub"}'
    grp.attrs["type"] = "blub"

    if cfg.normalize_acts:
        print("Computing training statistics ...")
        actions = np.concatenate(
            [grp[demo_key]["actions"] for demo_key in demo_keys["train"]]
        )
        stats = {
            "action": {
                "min": actions.min(axis=0),
                "max": actions.max(axis=0),
            }
        }

        pickle.dump(stats, open(os.path.join(hdf5_path, "stats"), 'wb'))

        print("Normalizing actions ...")
        for split in cfg.splits:
            for demo_key in demo_keys[split]:
                actions = grp[demo_key]["actions"]
                actions = normalize(actions, stats["action"])
                grp[demo_key]["actions"][...] = actions

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["blocking_control"] = cfg.blocking_control

    f.close()

    print("Saved at: {}".format(hdf5_path))


if __name__ == "__main__":
    run_experiment()
