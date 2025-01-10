import numpy as np
import argparse
import h5py

file_names = '/home/joel/projects/polymetis_franka/data/rummy/20240801_numpy/20240801_180924.npy'
data = np.load(file_names, allow_pickle=True)

def get_eef(robot_pose):
    actions_tmp = robot_pose.copy()
    actions_tmp[:-1, ..., :6] = (
        robot_pose[1:, ..., :6] - robot_pose[:-1, ..., :6]
    )
    actions = actions_tmp[:-1]
    return actions

# stack data
dic = {}
try:
    obs_keys = data[0].keys()
except:
    print(f"Error loading {file_names}")

for key in obs_keys:
    dic[key] = np.stack([d[key] for d in data])

left_pose = np.array(data[0]["action_left"])
right_pose = np.array(data[0]["action_right"])

left_action = get_eef(left_pose)
right_action = get_eef(right_pose)

# remove last state s_T
for key in obs_keys:
    dic[key] = dic[key][:-1]
    
demo_keys = {}
split = 'train'
demo_keys[split] = []

f = h5py.File("temp.hdf5", "w")

grp = f.create_group("data")
grp_mask = f.create_group("mask")

# create demo group
demo_key = f"demo_sample"
demo_keys[split].append(demo_key)
ep_data_grp = grp.create_group(demo_key)


#if cfg.data_source == "real":
    #actions[..., [3,4,5]] = actions[..., [4,3,5]]
    #actions[...,4] = -actions[...,4]
    #actions[...,3] = -actions[...,3]

# add action dataset
ep_data_grp.create_dataset("actions", data=left_action)

# add done dataset
dones = np.zeros(len(left_action)).astype(bool)
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
    if "color" in obs_key:
        # crop images for training
        # # resize images for training
        # obs = np.stack([cv2.resize(img, cfg.aug.camera_resize) for img in obs])
        # print(f"WARNING: replacing '{obs_key}' with 'front_rgb'!")
        obs_key = "front_rgb"
    if obs_key == "language_instruction":
        obs = np.array(obs, dtype='S80')

    ep_obs_grp.create_dataset(obs_key, data=obs)

ep_data_grp.attrs["num_samples"] = len(left_action)