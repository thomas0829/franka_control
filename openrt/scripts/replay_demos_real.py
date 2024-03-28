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
    cfg.robot.max_path_length = 100

    cfg.env.flatten = False

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    # TODO check if this makes a difference -> does when replaying action
    # env.action_space.low[:-1] = -1.0
    # env.action_space.high[:-1] = 1.0

    dataset_path = f"data/{cfg.exp_id}/train"
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
        img_obs = np.stack([obs["215122255213_rgb"] for obs in obss])
        imageio.mimsave(os.path.join(logdir, "replay.mp4"), img_obs)
        img_demo = np.stack([step["215122255213_rgb"] for step in episode])
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

        for i, (l,c) in enumerate(zip(labels, colors)):
            plt.plot(ee_act[:,i], color=c, label=f"{l} demo")
            plt.plot(ee_obs[:,i], color=c, linestyle="dashed", label=f"{l} replay")

        # labels = ["x", "y", "z", "roll", "pitch", "yaw"]
        # colors = ["tab:orange", "tab:blue", "tab:green", "red", "blue", "green"]

        # for i, (l,c) in enumerate(zip(labels, colors)):
        #     plt.plot(ee_act[:,i], color=c, label=f"{l} demo")
        #     plt.plot(ee_obs[:,i], color=c, linestyle="dashed", label=f"{l} replay")

        plt.legend()
        plt.show()
        
        time.sleep(5)

    env.reset()


if __name__ == "__main__":
    run_experiment()
