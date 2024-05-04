import glob
import os
import time

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from robot.robot_env import RobotEnv
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict


@hydra.main(
    config_path="../../configs/", config_name="collect_demos_real", version_base="1.1"
)
def run_experiment(cfg):

    logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    os.makedirs(logdir, exist_ok=True)

    cfg.robot.max_path_length = cfg.max_episode_length

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    camera_names = [k + "_rgb" for k in env.get_images().keys()]

    dataset_path = f"data/{cfg.exp_id}/{cfg.split}"
    file_names = glob.glob(f"{dataset_path}/episode_*.npy")
    assert len(file_names) > 0, f"WARNING: no data in {dataset_path}!"

    num_trajectories = 1

    for file in tqdm(file_names[:num_trajectories]):

        episode = np.load(file, allow_pickle=True)

        obs = env.reset()

        obss = []
        acts = []

        init_pos = env._curr_pos
        init_angle = env._curr_angle

        for step in tqdm(episode):

            start_time = time.time()

            act = step["action"]
            print(act)
            next_obs, rew, done, _ = env.step(act)

            obss.append(obs)
            acts.append(act)
            obs = next_obs

        env.reset()

        # visualize traj
        img_obs = np.stack([obs[camera_names[0]] for obs in obss])
        imageio.mimsave(os.path.join(logdir, "replay.mp4"), img_obs)
        # ugly hack to get the demo images
        img_demo = np.stack(
            [
                (
                    step[camera_names[0]]
                    if camera_names[0] in step.keys()
                    else step["front_rgb"]
                )
                for step in episode
            ]
        )
        imageio.mimsave(os.path.join(logdir, "demo.mp4"), img_demo)

        # plot difference between demo and replay
        plt.close()
        ee_obs = np.stack([obs["lowdim_ee"] for obs in obss])
        ee_act = []
        init_pos = env._curr_pos
        init_angle = env._curr_angle
        for step in episode:
            init_pos += step["action"][:3]
            init_angle += step["action"][3:6]
            ee_act.append(np.concatenate((init_pos.copy(), init_angle.copy())))
        ee_act = np.stack(ee_act)

        labels = ["x", "y", "z"]
        colors = ["tab:orange", "tab:blue", "tab:green"]

        for i, (l, c) in enumerate(zip(labels, colors)):
            plt.plot(ee_act[:, i], color=c, label=f"{l} demo")
            plt.plot(ee_obs[:, i], color=c, linestyle="dashed", label=f"{l} replay")

        plt.legend()
        plt.show()

    env.reset()


if __name__ == "__main__":
    run_experiment()
