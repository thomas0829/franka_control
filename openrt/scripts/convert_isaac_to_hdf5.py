import datetime
import glob
import os
import pickle
import cv2

import h5py
import hydra
import numpy as np
from tqdm import trange

import numpy as np
from scipy.spatial.transform import Rotation as R

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


@hydra.main(
    config_path="../../configs/", config_name="convert_demos_real", version_base="1.1"
)
def run_experiment(cfg):

    # REMOVE ME !!!!
    cfg.input_datasets = ["simpler_redcube_1000_seed_98"]
    cfg.data_dir = "/home/marius/Projects/polymetis_franka/data/"
    cfg.output_dataset = "simpler_redcube_1000_seed_98_blocking"
    # REMOVE ME !!!!

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
            file_names = glob.glob(os.path.join(dataset_path, split, "*.hdf5"))

            for i in trange(len(file_names)):

                # load data
                # data = np.load(file_names[i], allow_pickle=True)
                data = h5py.File(file_names[i],'r')

                world_offset_pos = np.array([0.2045, 0., 0.])
                ee_offset_euler = np.array([0., 0., -np.pi / 4])

                ee_pos = np.array(data["data"]["demo_0"]["obs"]["eef_pos"]).copy()
                # add pos offset
                ee_pos = ee_pos + world_offset_pos

                ee_quat = np.array(data["data"]["demo_0"]["obs"]["eef_quat"]).copy()
                # convert quat to euler
                ee_euler = quat_to_euler(ee_quat)
                # add angle offset
                ee_euler = add_angles(ee_offset_euler, ee_euler)

                qpos = np.array(data["data"]["demo_0"]["obs"]["joint_pos"]).copy()
                
                actions = np.array(data["data"]["demo_0"]["actions"]).copy()

                gripper = np.array(data["data"]["demo_0"]["obs"]["gripper_qpos"]).copy()
                
                # imgs
                world_img = np.array(data["data"]["demo_0"]["obs"]["world_camera_low_res_image"]).copy()
                wrist_img = np.array(data["data"]["demo_0"]["obs"]["hand_camera_low_res_image"]).copy()

                # convert gripper -1 close, 0 stay, +1 open to 0 open, 1 close
                prev_gripper_act = 1
                gripper_act = actions[..., -1].copy()
                # sim runs continuous grasp -> only grasp when fully closed to match blocking
                if cfg.blocking_control:
                    not_quite_done_yet = np.where(np.abs(gripper[1:,0] - gripper[:-1,0]) > 1e-3)
                    gripper_act[not_quite_done_yet] = 0
                for i in range(len(gripper_act)):
                    gripper_act[i] = prev_gripper_act if gripper_act[i] == 0 else gripper_act[i]
                    prev_gripper_act = gripper_act[i]
                gripper_act[np.where(gripper_act == 1)] = 0
                gripper_act[np.where(gripper_act == -1)] = 1
                actions[..., -1] = gripper_act
                
                lowdim_gripper = np.sum(gripper, axis=1)[:,None]

                dic = {
                    "lowdim_ee": np.concatenate((ee_pos, ee_euler, lowdim_gripper), axis=1),
                    "lowdim_qpos": np.concatenate((qpos, lowdim_gripper), axis=1),
                    "front_rgb": world_img,
                    "wrist_rgb": wrist_img
                }

                obs_keys = dic.keys()
                
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

                # create demo group
                demo_key = f"demo_{episodes}"
                demo_keys[split].append(demo_key)
                ep_data_grp = grp.create_group(demo_key)

                # compute shortest angle -> avoid wrap around
                actions[..., 3:6] = shortest_angle(actions[..., 3:6])

                # add action dataset
                ep_data_grp.create_dataset("actions", data=actions)

                # add done dataset
                dones = np.zeros(len(actions)).astype(bool)
                dones[-1] = True
                ep_data_grp.create_dataset("dones", data=dones)

                # create obs and next_obs groups
                ep_obs_grp = ep_data_grp.create_group("obs")

                # add obs and next_obs datasets
                for obs_key in obs_keys:
                    obs = dic[obs_key]
                    if obs_key == "language_instruction":
                        continue
                    if "_rgb" in obs_key:
                        # crop images for training
                        x_min, x_max, y_min, y_max = cfg.aug.camera_crop
                        obs = obs[:, x_min : x_max, y_min : y_max]
                        # resize images for training
                        obs = np.stack([cv2.resize(img, cfg.aug.camera_resize) for img in obs])

                    ep_obs_grp.create_dataset(obs_key, data=obs)

                ep_data_grp.attrs["num_samples"] = len(actions)

                episodes += 1

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
