import glob
import os
import time

from pathlib import Path
import sys
#sys.path.append("/home/marius/Documents/Entong/robomimic2")
sys.path.append("/home/joel/projects/robomimic_openrt")

import hydra

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from robot.wrappers.crop_wrapper import CropImageWrapper
from robot.wrappers.resize_wrapper import ResizeImageWrapper
from robot.robot_env import RobotEnv
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict
import cv2
import robomimic.utils.file_utils as FileUtils
import h5py
# from openrt.scripts.convert_np_to_hdf5 import augmentation_pcd
#sys.path.append("/home/marius/Documents/Entong/robosuite")
sys.path.append("/home/joel/projects/robosuite")
from robosuite.utils.camera_utils import *
import argparse
import imageio
from pathlib import Path

def unnormalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 0.5 * (arr + 1) * (max_val - min_val) + min_val

@hydra.main(config_path="../../configs/",
            config_name="collect_demos_real.yaml",
            version_base="1.1")

def run_experiment(cfg):
    mode = "test" #["train", "test", "replay", "train_obs"]
    use_pc = False
    num_pc = 1024
    use_rgb= False
    use_real = True
    device = "cuda:0"
    cam_name = cfg.cam_name
    trial_num = cfg.trial_num

    model_path = f"{cfg.ckpt_path}/{os.listdir(cfg.ckpt_path)[0]}/models"
    print(f'model path: {model_path}')
    if cfg.eval_last:
        import re 
        filenames = os.listdir(model_path)
        pattern = re.compile(r'model_epoch_(\d+)\.pth')
        # Extract numbers and convert them to integers
        numbers = [int(pattern.search(filename).group(1)) for filename in filenames]
        numbers.sort()
        epoch_num = numbers[-1]
        ckpt_path = f"{model_path}/model_epoch_{epoch_num}.pth"
    else:
        if cfg.epoch_num is None:
            raise Exception("Please provide an epoch number since you are not evaluating the last epoch")
        else:
            epoch_num = cfg.epoch_num
            ckpt_path = f"{model_path}/model_epoch_{epoch_num}.pth"
    
    file = h5py.File(f"{cfg.base_dir}/{cfg.exp_id}/demos.hdf5", 'r')
    dataset = file["data"]
    import pickle
    with open(f"{cfg.base_dir}/{cfg.exp_id}/stats", 'rb') as file:
        # Load data from the file using pickle
        stats = pickle.load(file)["action"]

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path,
                                                         device=device,
                                                         verbose=True)
   
    cfg.robot.blocking_control = True

    if cfg.robot.blocking_control:
        cfg.robot.control_hz = cfg.control_hz

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        #env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    #camera_names = env.unwrapped._robot.camera_names.copy()
    camera_names = ["215122255213"]

    if cfg.aug.camera_crop is not None:
        env = CropImageWrapper(
            env,
            x_min=cfg.aug.camera_crop[0],
            x_max=cfg.aug.camera_crop[1],
            y_min=cfg.aug.camera_crop[2],
            y_max=cfg.aug.camera_crop[3],
            image_keys=[cn + "_rgb" for cn in camera_names],
            crop_render=True,
        )

    #if cfg.aug.camera_resize is not None:
    #    env = ResizeImageWrapper(
    #        env,
    #        size=cfg.aug.camera_resize,
    #        image_keys=[cn + "_rgb" for cn in camera_names],
    #    )
   
        
    import pdb

    #pdb.set_trace()
    count = 0
    num_traj = 0 # first traj in the hdf5 file
       
    # reset env
    obs = env.reset()
    
    # original trajectory
    traj0 = np.array(dataset[f"demo_{num_traj}"]["actions"])
    # normalize trajectory
    traj0 = unnormalize(traj0, stats=stats)
    observations0 = dataset[f"demo_{num_traj}"]["obs"]
    buffer = []
    obs_buffer = {}
    for key in [f"215122255213_rgb","lowdim_qpos","lowdim_ee"]:
        if key == f"215122255213_rgb":
            image = np.transpose(observations0[f"front_rgb"][0], (2, 0, 1)) / 255
            obs_buffer[key] = np.concatenate([image[None], image[None]],axis=0)
        else:
            obs_buffer[key] = np.concatenate([observations0[key][0][None], observations0[key][0][None]],axis=0)
    #for i in range(len(traj0)-1):
    print(f'Doing trial number {trial_num}')
    
    policy.start_episode()
    for i in range(cfg.traj_length):
        print(f'Step: {i}')

        start_time = time.time()

        # if use_rgb:
        obs["215122255213_rgb"] = np.transpose(obs["215122255213_rgb"], (2, 0, 1)) / 255

        pcd_obs = {}
        
        
        for key in obs.keys():
            pcd_obs[key] = obs[key][None]

        if mode == "train" or mode == "test" :
            
            if use_pc:
                pc = augmentation_pcd(pcd_obs)
                # visualize_pcd(pc[0])
                obs["pcd"] = np.transpose(pc[0], (1, 0))
            
            if '_dp' in ckpt_path:
                for key in [f"215122255213_rgb","lowdim_qpos","lowdim_ee"]:
                    obs_buffer[key][0] = obs_buffer[key][-1].copy()
                    obs_buffer[key][1] = obs[key].copy()
                obs_buffer[f"front_rgb"] = obs_buffer["215122255213_rgb"]
                act = policy(obs_buffer)
            else:
                obs[f"front_rgb"] = obs["215122255213_rgb"]
                obs.pop("215122255213_depth") # this causes error if not popped
                act = policy(obs)
            act = unnormalize(act, stats=stats)
            
        elif mode=="train_obs":
            
            obs0 = {}
            
            for key in observations0.keys():
                obs0[key] = observations0[key][i]

                if key == "pcd":
                    obs0["pcd"] = np.transpose(obs0[key], (1, 0))
            if use_real:
                
                pre_obs0 = {}
            
                pre_obs0[f"{cam_name}_rgb"] = np.transpose(obs0[f"{cam_name}_rgb"], (2, 0, 1)) / 255
                pre_obs0["lowdim_qpos"] = obs0["lowdim_qpos"]
                pre_obs0["lowdim_ee"] = obs0["lowdim_ee"]

            act = policy(pre_obs0)
            act = unnormalize(act, stats=stats)
        elif mode == "replay":
            
            obs0 = {}
            
            for key in observations0.keys():
                obs0[key] = observations0[key][i]

                if key == "pcd":
                    obs0["pcd"] = np.transpose(obs0[key], (1, 0))
            if use_real:
                
                pre_obs0 = {}
            
                pre_obs0[f"{cam_name}_rgb"] = np.transpose(obs0[f"{cam_name}_rgb"], (2, 0, 1)) / 255
                pre_obs0["lowdim_qpos"] = obs0["lowdim_qpos"]
                pre_obs0["lowdim_ee"] = obs0["lowdim_ee"]

            act = traj0[i]
        
        
        act[-1] = np.round(act[-1])

        next_obs, rew, done, _ = env.step(act)         
        image = next_obs["215122255213_rgb"]
        
        if mode=="train_obs":
            image  = np.concatenate([image,obs0[f"{cam_name}_rgb"]],axis=1)

        #cv2.imshow('Real-time video', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        if cfg.save_info:
            directory_path = f"{cfg.base_dir}/{cfg.exp_id}/eval/{cfg.ckpt_path.split('/')[-1]}/epoch_{epoch_num}/{trial_num}_ood"
            if not os.path.exists(directory_path):
                # Create the directory
                os.makedirs(directory_path)
                print(f"Directory '{directory_path}' created.")
            cv2.imwrite(f"{directory_path}/{count}.png",cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
        obs = next_obs
        count += 1

    env.reset()

import argparse

if __name__ == "__main__":
    run_experiment()
