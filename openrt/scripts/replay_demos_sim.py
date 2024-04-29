import glob
import os
import time

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from robot.crop_wrapper import CropImageWrapper
from robot.resize_wrapper import ResizeImageWrapper
from robot.robot_env import RobotEnv
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict


@hydra.main(
    config_path="../../configs/", config_name="collect_cube_sim", version_base="1.1"
)
def run_experiment(cfg):

    logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    os.makedirs(logdir, exist_ok=True)

    cfg.robot.blocking_control = True

    dataset_path = f"data/{cfg.exp_id}/train"
    file_names = glob.glob(f"{dataset_path}/episode_*.npy")
    assert len(file_names) > 0, f"WARNING: no data in {dataset_path}!"

    num_trajectories = 10

    for i in trange(num_trajectories):
    
        episode = np.load(file_names[i], allow_pickle=True)

        cfg.robot.max_path_length = len(episode)

        env = make_env(
            robot_cfg_dict=hydra_to_dict(cfg.robot),
            env_cfg_dict=hydra_to_dict(cfg.env),
            seed=cfg.seed,
            device_id=0,
            verbose=True,
        )

        camera_names = env.unwrapped._robot.camera_names.copy()

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

        obs = env.reset()

        obss = []
        acts = []

        for step in tqdm(episode):
            
            start_time = time.time()
            
            act = step["action"]
            print(act)
            next_obs, rew, done, _ = env.step(act)

            # init_pos += act[:3]
            # init_angle += act[3:6]
            # gripper = act[6]

            # env._update_robot(
            #     np.concatenate((init_pos, init_angle, [gripper])),
            #     action_space="cartesian_position", blocking=True,
            # )
            # next_obs = env.get_observation()
            
            obss.append(obs)
            acts.append(act)
            obs = next_obs

            comp_time = time.time() - start_time
            sleep_left = max(0, (1 / cfg.robot.control_hz) - comp_time)
            # time.sleep(sleep_left)
            # time.sleep(1/cfg.robot.control_hz)

        env.reset()

        # visualize traj
        img_obs = np.stack([obs["front_rgb"] for obs in obss])
        img_demo = np.stack([step["front_rgb"] for step in episode])
        imageio.mimsave(os.path.join(logdir, f"demo_replay_{i}.mp4"), np.concatenate((img_demo, img_obs), axis=2))

        # plot difference between demo and replay
        ee_obs = np.stack([obs["lowdim_ee"] for obs in obss])
        ee_act = np.stack([obs["lowdim_ee"][:6] for obs in obss])

        labels = ["x", "y", "z", "r", "p", "y"]
        colors = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        n = 3
        for j, (l,c) in enumerate(zip(labels[:n], colors[:n])):
            plt.plot(ee_act[:,j], color=c, label=f"{l} demo")
            plt.plot(ee_obs[:,j], color=c, linestyle="dashed", label=f"{l} replay")
        plt.legend()
        plt.savefig(os.path.join(logdir, f"poses_{i}.png"))
        plt.close()
    
    env.reset()


if __name__ == "__main__":
    run_experiment()
