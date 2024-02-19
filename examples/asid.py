import argparse
import copy
import datetime
import os
import time

import imageio
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from asid.wrapper.asid_vec import make_vec_env
from robot.sim.vec_env.vec_wrapper import SubVecEnv
from utils.logger import Video, configure_logger


class LoggerCallback(BaseCallback):
    """
    A custom callback that derives from BaseCallback.
    """

    def __init__(
        self, cfg, eval_interval, save_dir, save_interval, seed=0, gpu_id=0, verbose=0
    ):
        super(LoggerCallback, self).__init__(verbose)

        self.eval_interval = eval_interval

        self.save_dir = save_dir
        self.save_interval = save_interval

        self.num_env = 8
        self.p_envs = make_vec_env(
            cfg,
            num_workers=num_workers,
            seed=seed,
            device_id=gpu_id,
            exp_reward=True,
        )
        self.p_envs.reset()
        self.reward_sum = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        self.reward_sum += np.mean(self.locals["rewards"])

        if np.mod(self.n_calls, 20) == 0:
            self.logger.record("return", self.reward_sum)
            self.reward_sum = 0

        if (self.n_calls - 1) % self.eval_interval == 0:
            evaluate(self.model, self.p_envs, self.logger, tag="train")
            self.compute_dev()

        if (self.n_calls - 1) % self.save_interval == 0:
            self.model.save(self.save_dir + "_step_" + str(self.n_calls))

    def compute_dev(self):
        total_reward = 0
        all_dev = np.zeros(self.num_env)
        obs = self.p_envs.reset()
        init_obs_pos = []
        for i in range(self.num_env):
            init_obs_pos.append(copy.deepcopy(obs[i][2:4]))
        dones = False
        while not np.all(dones):
            actions, _state = self.locals["self"].predict(obs)
            next_obs, rewards, dones, infos = self.p_envs.step(actions)
            obs = next_obs
            total_reward += np.sum(rewards)
            for i in range(self.num_env):
                all_dev[i] += np.linalg.norm(init_obs_pos[i] - obs[i][2:4]) ** 2
        self.logger.record("current_policy_value", total_reward / self.num_env)
        self.logger.record("total_obj_dev", np.mean(all_dev))


def evaluate(policy, eval_envs, logger, tag="eval"):

    render = torch.cuda.device_count() > 0
    eval_envs.render_mode = "rgb_array"

    obs = eval_envs.reset()
    dones, infos, frames = False, [], []

    episode_returns = []
    episode_successes = []

    # Assume all eval_envs terminate at the same time
    while not np.all(dones):
        # Select action
        actions, _state = policy.predict(obs, deterministic=False)
        # Take environment step
        next_obs, rewards, dones, infos = eval_envs.step(actions)
        episode_returns.append(rewards)
        episode_successes.append([info["is_success"] for info in infos])

        if render:
            frames.append(eval_envs.render())
        obs = next_obs

    # Record episode statistics
    avg_return = np.mean(np.sum(np.stack(episode_returns), axis=0))
    avg_success = np.mean(np.any(np.stack(episode_successes), axis=0))
    logger.record(f"{tag}/return", avg_return)
    logger.record(f"{tag}/success", avg_success)

    # Record videos
    if render:
        # -> (b, t, 3, h, w)
        video = np.stack(frames).transpose(1, 0, 4, 2, 3)
        logger.record(
            f"{tag}/trajectory/env/camera",
            Video(video, fps=30),
            exclude=["stdout"],
        )
    logger.dump(step=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--exp", type=str)
    parser.add_argument("--save_dir", type=str, default="data")

    # hardware
    parser.add_argument("--dof", type=int, default=2, choices=[2, 3, 4, 6])
    parser.add_argument(
        "--robot_type", type=str, default="panda", choices=["panda", "fr3"]
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        default=None,
        choices=[None, "localhost", "172.16.0.1"],
    )
    parser.add_argument(
        "--camera_model", type=str, default="realsense", choices=["realsense", "zed"]
    )

    # training
    parser.add_argument("--max_episode_length", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    args.exp = "test"
    assert args.exp is not None, "Specify --exp"
    device = torch.device(
        ("cuda:" + str(args.gpu_id))
        if args.gpu_id >= 0.0 and torch.cuda.is_available()
        else "cpu"
    )

    control_hz = 10

    cfg = {
        "control_hz": control_hz,
        "DoF": args.dof,
        "robot_type": args.robot_type,
        "gripper": False,
        "ip_address": args.ip_address,
        "camera_model": args.camera_model,
        "camera_resolution": (480, 480),
        "imgs": False,
        "max_path_length": args.max_episode_length,
        "model_name": "rod_franka",
        "on_screen_rendering": False,
    }

    num_workers = 2

    env = make_vec_env(
        cfg,
        num_workers=num_workers,
        seed=0,
        device_id=args.gpu_id,
        exp_reward=True,
    )

    from utils.experiment import setup_wandb

    setup_wandb(cfg, name=args.exp, entity="memmelma", project="asid_re")

    logdir = "logdir"
    ckptdir = logdir + "/checkpoints"
    logger = configure_logger(logdir, ["wandb"])

    import stable_baselines3 as sb3
    from stable_baselines3 import SAC

    act_noise = sb3.common.noise.NormalActionNoise(
        np.zeros(args.dof), np.ones(args.dof), decay=0.9999993
    )
    act_noise = sb3.common.noise.VectorizedActionNoise(act_noise, num_workers)

    model = SAC(
        "MlpPolicy",
        env,
        device=device,
        learning_starts=500,
        ent_coef=0.0001,
        train_freq=1,
        gradient_steps=1,
        action_noise=act_noise,
    )

    eval_interval = 1
    save_interval = 100
    callback = LoggerCallback(
        cfg, eval_interval=eval_interval, save_dir=ckptdir, save_interval=save_interval
    )

    # train
    # if not cfg.train.load_policy:
    model.learn(total_timesteps=1e6, callback=callback, progress_bar=True)
    model.save(ckptdir)
    env.close()
