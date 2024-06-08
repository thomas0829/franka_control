import collections
import json
import os
import time
import pickle
from dataclasses import dataclass

import cv2
import gym
import h5py
import hydra
import imageio
import matplotlib.pyplot as plt

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import numpy as np
import torch
from tqdm import trange
import sys
sys.path.append('/home/marius/Projects/polymetis_franka/training/robomimic')
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robot.wrappers.crop_wrapper import CropImageWrapper
from robot.wrappers.resize_wrapper import ResizeImageWrapper
from robot.sim.vec_env.vec_env import make_env
#from training.weird_bc.models.policies import MixedGaussianPolicy
from training.weird_bc.train_script import plot_trajectory
from openrt.scripts.convert_np_to_hdf5 import normalize, unnormalize
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Image, Video, configure_logger
from utils.system import get_device, set_gpu_mode
# from PIL import Image
from robomimic.utils.lang_utils import get_lang_emb
from robot.sim.mujoco.distractor_wrapper import DistWrapper

def make_multi_cube_env(cfg, color_ids=[0, 1, 2], savedir=None):

    colors = [
        [0., 0.9, 0., 1.],
        [0.9, 0., 0., 1.],
        [0., 0., 0.9, 1.]
    ]
    color_names = [
        "green",
        "red",
        "blue"
    ]
    offsets = [0.1, -0.1]

    language_instruction = f"pick up the {color_names[color_ids[0]]} cube"

    robot_cfg_dict = hydra_to_dict(cfg.robot)
    robot_cfg_dict["model_name"] = "two_cubes_franka"

    env_cfg_dict = hydra_to_dict(cfg.env)
    env_cfg_dict["obj_id"] = "cube"
    env_cfg_dict["obj_rgba"] = colors[color_ids[0]]
    # env_cfg_dict["obj_pose_init"][1] = offsets[color_ids[0]]
    # env_cfg_dict["obj_pose_noise_dict"] = None
    # env_cfg_dict["obj_pose_noise_dict"] = {
    #     "x": { "min": -0.1, "max": 0.1 },
    #     "y": { "min": -0.1, "max": 0.1 },
    #     # "yaw": { "min": -0., "max": 0.0 },
    #     "yaw": { "min": -0.785, "max": 0.785 },
    # }
    env = make_env(
            robot_cfg_dict=robot_cfg_dict,
            env_cfg_dict=env_cfg_dict,
            seed=cfg.seed,
            device_id=0,
            verbose=True,
        )

    dis_cfg_dict = hydra_to_dict(cfg.env)
    dis_cfg_dict["reset_data_on_reset"] = False
    dis_cfg_dict["obj_id"] = "distractor_0"
    dis_cfg_dict["obj_rgba"] = colors[color_ids[1]]
    # dis_cfg_dict["obj_pose_init"][1] = offsets[color_ids[1]]
    # dis_cfg_dict["obj_pose_noise_dict"] = None
    # env_cfg_dict["obj_pose_noise_dict"] = {
    #     "x": { "min": -0.1, "max": 0.1 },
    #     "y": { "min": -0.1, "max": 0.1 },
    #     # "yaw": { "min": -0., "max": 0.0 },
    #     "yaw": { "min": -0.785, "max": 0.785 },
    # }
    env = DistWrapper(env, **dis_cfg_dict)

    return env, language_instruction

@hydra.main(
    version_base=None, config_path="../../configs/", config_name="eval_robomimic_multi_sim"
)
def run_experiment(cfg):
    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[{cfg.seed}][eval]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    set_random_seed(cfg.seed)

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed))
    logger = configure_logger(logdir, cfg.log.format_strings)

    subdir = "open_loop" if cfg.open_loop else "closed_loop"
    subdir += "_" + cfg.open_loop_split if cfg.open_loop else ""
    os.makedirs(os.path.join(logdir, subdir), exist_ok=True)

    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    # load dataset for open loop execution
    if cfg.open_loop:
        data = h5py.File(
            cfg.data_path+"/demos.hdf5",
            "r",
            swmr=True,
            libver="latest",
        )
        demo_keys = np.asarray(data["mask"][cfg.open_loop_split])
        print("Data loaded [robomimic].")

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # load policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=cfg.ckpt_path,
        device=device,
        verbose=True,
    )
    print("Pretrained weights loaded [robomimic].")

    # load stats to normalize actions
    stats = pickle.load(open(os.path.join(cfg.data_path, "stats"), "rb"))
    print("Data stats loaded [robomimic].")

    # create env
    # env = make_env(
    #     robot_cfg_dict=hydra_to_dict(cfg.robot),
    #     env_cfg_dict=(
    #         None if cfg.robot.ip_address is not None else hydra_to_dict(cfg.env)
    #     ),
    #     seed=cfg.seed,
    #     device_id=0,
    #     verbose=True,
    # )

    cfg.robot.visual_dr = False

    # color #0 for first half of demos
    env, language_instruction = make_multi_cube_env(cfg, color_ids=[0, 1, 2])
    print(language_instruction)
    # color #1 for second half of demos
    # env, language_instruction = make_multi_cube_env(cfg, color_ids=[1, 0, 2])
    # color #2 for distractor

    if cfg.robot.ip_address is not None:
        assert env._num_cameras > 0, "ERROR: not camera(s) connected!"
    
    # camera_names = [k for k in env.get_images().keys()] if cfg.robot.ip_address is not None else env.unwrapped._robot.camera_names.copy()
    # get training camera names from config
    config = json.loads(ckpt_dict["config"])
    camera_names = [
        cn.replace("_rgb", "")
        for cn in config["observation"]["modalities"]["obs"]["rgb"]
    ]
    # camera_names = ["215122255213"]
   

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

    if cfg.aug.camera_resize is not None:
        env = ResizeImageWrapper(
            env,
            size=cfg.aug.camera_resize,
            image_keys=[cn + "_rgb" for cn in camera_names],
        )

    lang_embed = get_lang_emb(language_instruction)

    obj_poses = []

    # open loop
    if cfg.open_loop:
        n_rollouts = min(len(demo_keys), cfg.n_rollouts)
    else:
        n_rollouts = cfg.n_rollouts

    for i in trange(n_rollouts):
      

        policy.start_episode()

        obs = env.reset()

        imgs = []
        acts = []
        obss = []

        # load eval traj
        if cfg.open_loop:
            eval_traj = data["data"][demo_keys[i].astype(str)]
            cfg.robot.max_path_length = len(eval_traj["obs"]["action"])

        errors = []
        for j in trange(cfg.robot.max_path_length):

            # save image and resize
            if len(camera_names):
                img_tmp = obs[camera_names[0] + "_rgb"]
            else:
                img_tmp = env.render()
            img_resize = cv2.resize(img_tmp, dsize=cfg.aug.camera_resize)
            # img_temp = Image.fromarray(img_resize)
            # img_temp.save('temp.jpg')
            imgs.append(img_resize.transpose(2, 0, 1)[None])

            obs_original = obs.copy()

            # replace obs with eval traj
            if cfg.open_loop:
                
                if j == 0:
                    print(eval_traj["obs"]["language_instruction"][i])

                obs = {
                    "lowdim_ee": eval_traj["obs"]["lowdim_ee"][j],
                    "lowdim_qpos": eval_traj["obs"]["lowdim_qpos"][j],
                    # "front_rgb": eval_traj["obs"]["215122255213_rgb"][j]
                    "front_rgb": eval_traj["obs"]["front_rgb"][j],
                    "lang_embed": eval_traj["obs"]["lang_embed"][j]
                }
             
            # preprocess imgs
            for key in [cn + "_rgb" for cn in camera_names]:
                obs[key] = obs[key].transpose(2, 0, 1)

            with torch.no_grad():
                if "lang_embed" not in obs.keys():
                    obs["lang_embed"] = lang_embed

                # remove depth from observations
                for cn in camera_names:
                    if cn + "_depth" in obs:
                        del obs[cn + "_depth"]
               
                # obs["front_rgb"] = obs["215122255213_rgb"]#/255
                
                # run policy and normalize actions
                act = policy(ob=obs)
                act = unnormalize(act, stats["action"])
                       
            obs["front_rgb"] = env.render().transpose(2, 0, 1)
           
            # binarize gripper
            act[-1] = 1 if act[-1] > 0.5 else 0

            acts.append(act)
            obs_original["action"] = act
            obss.append(obs_original)
            # print("applied act", np.around(act, 3))

            # step env
            obs, reward, done, _ = env.step(act)
           
            # compute error for open loop predictions
            if cfg.open_loop:
                gt_act = unnormalize(eval_traj["actions"][j], stats["action"])
                error = np.around(np.sum(np.abs(gt_act[:6] - act[:6])), 3)
                errors.append(error)
        if cfg.open_loop:
            print(
                f"episode {i} | sum abs error {np.sum(errors)}"  #  | act {np.around(act,3)} | gt {np.around(gt_act,3)}",
                # f"episode {i} | step {j} | abs error {error}"  #  | act {np.around(act,3)} | gt {np.around(gt_act,3)}",
            )

        # dump rollout
        np.save(os.path.join(logdir, subdir, f"eval_episode_{i}.npy"), obss)

        # obj poses for success check
        if cfg.robot.ip_address is None:
            obj_poses.append(obs["obj_pose"])

        # T,C,H,W
        video = np.stack(imgs)[:, 0]
        if cfg.open_loop:
            video = np.concatenate((video, np.array(eval_traj["obs"]["front_rgb"]).transpose(0,3,1,2)), axis=2)

        # save trajectory plot -> takes T,C,H,W
        plot_img = plot_trajectory(
            pred_actions=np.stack(acts),
            true_actions=eval_traj["obs"]["action"] if cfg.open_loop else None,
            imgs=video,
        )
        logger.record(
            f"images/eval_{i}",
            Image(plot_img, dataformats="HWC"),
            exclude=["stdout"],
        )
        plt.imsave(
            os.path.join(logdir, subdir, f"eval_{i}_plot.png"),
            plot_img,
        )

        # save video local -> takes T,H,W,C
        imageio.mimsave(
            os.path.join(logdir, subdir, f"eval_{i}.mp4"), video.transpose(0, 2, 3, 1)
        )
        # save video wandb -> takes T,C,H,W
        logger.record(
            f"videos/eval_{i}",
            Video(video, fps=20),  # duration=1000 * 1 / 20),
            exclude=["stdout"],
        )

    logger.dump()
    if cfg.open_loop:
        data.close()


if __name__ == "__main__":
    run_experiment()
