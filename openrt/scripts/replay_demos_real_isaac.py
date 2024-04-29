import h5py

import os
import time
import numpy as np
from tqdm import tqdm
import imageio
import hydra
import glob
import matplotlib.pyplot as plt

from robot.robot_env import RobotEnv
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict


@hydra.main(
    config_path="../configs/", config_name="collect_cube_real", version_base="1.1"
)
def run_experiment(cfg):

    logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    os.makedirs(logdir, exist_ok=True)

    cfg.robot.max_path_length = cfg.max_episode_length

    cfg.robot.DoF = 6
    cfg.robot.control_hz = 1
    cfg.robot.gripper = True
    cfg.robot.blocking_control = True
    cfg.robot.on_screen_rendering = False

    cfg.env.flatten = False

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    env.action_space.low[:-1] = -1.0
    env.action_space.high[:-1] = 1.0

    env._reset_joint_qpos = np.array([0.0, -0.8, 0.0, -2.3, 0.1, 2.3, 0.8])

    dataset_path = "data/helen_504/train"
    # dataset_path = f"data/{cfg.exp_id}/train"
    file_names = glob.glob(f"{dataset_path}/*.hdf5")
    assert len(file_names) > 0, f"WARNING: no data in {dataset_path}!"

    num_trajectories = 1

    for file in tqdm(file_names[:num_trajectories]):

        episode = h5py.File(file, "r+")["data"]["demo_0"]

        # data.append(
        #     {
        #         'actions': demo['actions'],
        #         'obs': demo['obs'],
        #         "images": demo['obs']['world_camera_low_res_image']
        #     }
        # )

        # cfg.robot.blocking_control = False

        obs = env.reset()

        # des_pose = env.get_observation()["lowdim_ee"].copy()
        # # des_pose[5] -= np.pi / 4
        # error = des_pose - env.get_observation()["lowdim_ee"]

        # while np.sum(error) > 1e-3:
        #     env.step(error)
        #     error = des_pose - env.get_observation()["lowdim_ee"]

        # cfg.robot.blocking_control = True

        # act = np.zeros(7)
        # act[5] = -np.pi / 4
        # env.step(act)

        obss = []
        acts = []

        actions = episode["actions"]

        # unnormalize actions
        actions[:, :6] *= np.array(
            [[0.05, 0.05, 0.05, 0.17453293, 0.17453293, 0.17453293]]
        )

        for act in tqdm(actions):

            print(act)
            act[3:] = 0.
            next_obs, rew, done, _ = env.step(act)

            obss.append(obs)
            acts.append(act)
            obs = next_obs

        env.reset()

        # visualize traj
        img_obs = np.stack([obs["215122255213_rgb"] for obs in obss])
        imageio.mimsave(os.path.join(logdir, "replay.mp4"), img_obs)
        # ugly hack to get the demo images

        img_demo = episode["obs"]["world_camera_low_res_image"]
        imageio.mimsave(os.path.join(logdir, "demo.mp4"), img_demo)

        # plot difference between demo and replay
        plt.close()
        ee_obs = np.stack([obs["lowdim_ee"] for obs in obss])
        ee_act = episode["obs"]["eef_pos"]

        labels = ["x", "y", "z"]
        colors = ["tab:orange", "tab:blue", "tab:green"]

        for i, (l, c) in enumerate(zip(labels, colors)):
            plt.plot(ee_act[:, i], color=c, label=f"{l} demo")
            plt.plot(ee_obs[:, i], color=c, linestyle="dashed", label=f"{l} replay")

        plt.legend()
        plt.show()

        time.sleep(5)

    env.reset()


if __name__ == "__main__":
    run_experiment()
