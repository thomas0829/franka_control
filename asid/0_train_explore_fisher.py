import os

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import copy

import hydra
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from asid.wrapper.sim.asid_vec import make_env, make_vec_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Video, configure_logger
from utils.system import get_device, set_gpu_mode


class LoggerCallback(BaseCallback):
    """
    A custom callback that derives from BaseCallback.
    """

    def __init__(
        self, eval_envs, eval_interval, save_dir, save_interval, verbose=False
    ):
        super(LoggerCallback, self).__init__(verbose)

        self.eval_interval = eval_interval

        self.save_dir = save_dir
        self.save_interval = save_interval

        self.eval_envs = eval_envs
        self.eval_envs.reset()

        self.reward_sum = 0.0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        self.reward_sum += np.mean(self.locals["rewards"])

        if np.all(self.locals["dones"]):
            self.logger.record("return", self.reward_sum)
            self.reward_sum = 0

        if (self.n_calls - 1) % self.eval_interval == 0:
            evaluate(self.model, self.eval_envs, self.logger, tag="train")
            self.compute_obj_dev()

        if (self.n_calls - 1) % self.save_interval == 0:
            self.model.save(self.save_dir + "_step_" + str(self.n_calls))

    def compute_obj_dev(self):
        """
        This methods compute the deviation (L2) of the object pose from its initial pose.
        """
        total_reward = 0
        all_dev = np.zeros(self.eval_envs.num_envs)

        obs = self.eval_envs.reset()
        init_obj_poses = self.eval_envs.get_obj_pose()

        dones = False
        while not np.all(dones):
            actions, _state = self.model.predict(obs)
            next_obs, rewards, dones, infos = self.eval_envs.step(actions)
            obs = next_obs

            total_reward += np.sum(rewards)
            for i in range(self.eval_envs.num_envs):
                all_dev[i] += (
                    np.linalg.norm(init_obj_poses[i] - self.eval_envs.get_obj_pose()[i])
                    ** 2
                )

        self.logger.record(
            "current_policy_value", total_reward / self.eval_envs.num_envs
        )
        self.logger.record("total_obj_dev", np.mean(all_dev))


def evaluate(policy, eval_envs, logger, tag="eval"):
    render = torch.cuda.device_count() > 0
    eval_envs.render_mode = "rgb_array"

    obs = eval_envs.reset()
    dones, infos, frames = False, [], []

    episode_returns = []
    # episode_successes = []

    # Assume all eval_envs terminate at the same time
    while not np.all(dones):
        # Select action
        actions, _state = policy.predict(obs, deterministic=False)
        # Take environment step
        next_obs, rewards, dones, infos = eval_envs.step(actions)
        episode_returns.append(rewards)
        # episode_successes.append([info["episode_success"] for info in infos])

        if render:
            frames.append(eval_envs.render())
        obs = next_obs

    # Record episode statistics
    avg_return = np.mean(np.sum(np.stack(episode_returns), axis=0))
    logger.record(f"{tag}/return", avg_return)
    # avg_success = np.mean(np.any(np.stack(episode_successes), axis=0))
    # logger.record(f"{tag}/success", avg_success)

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


@hydra.main(config_path="../configs/", config_name="asid", version_base="1.1")
def run_experiment(cfg):
    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[explore][{cfg.seed}]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    set_random_seed(cfg.seed)
    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "explore")
    logger = configure_logger(logdir, cfg.log.format_strings)

    # train env
    envs = make_vec_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        device_id=0,
        exp_reward=True,
        delta=cfg.train.fd_delta,
        normalization=cfg.train.rew_normalization,
    )

    # algorithm
    act_noise = sb3.common.noise.NormalActionNoise(
        np.zeros(envs.action_space.shape),
        np.ones(envs.action_space.shape),
        decay=0.9999993,
    )
    act_noise = sb3.common.noise.VectorizedActionNoise(act_noise, cfg.num_workers)
    model = SAC(
        "MlpPolicy",
        envs,
        device=device,
        learning_starts=cfg.train.algorithm.learning_starts,
        ent_coef=cfg.train.algorithm.ent_coef,
        train_freq=cfg.train.algorithm.train_freq,
        gradient_steps=cfg.train.algorithm.gradient_steps,
        action_noise=act_noise,
    )

    # eval env
    eval_envs = make_vec_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        num_workers=cfg.num_workers_eval,
        seed=cfg.seed + 100,
        device_id=0,
        exp_reward=True,
        delta=cfg.train.fd_delta,
        normalization=cfg.train.rew_normalization,
    )

    # set logger
    model.set_logger(logger)
    ckptdir = os.path.join(logdir, "policy")

    callback = LoggerCallback(
        eval_envs=eval_envs,
        eval_interval=cfg.log.eval_interval,
        save_dir=ckptdir,
        save_interval=cfg.log.save_interval,
    )

    # train
    if not cfg.train.load_policy:
        model.learn(
            total_timesteps=cfg.train.total_timesteps,
            callback=callback,
            progress_bar=True,
        )
        model.save(ckptdir)
    envs.close()

    model = model.load(ckptdir)

    # eval
    evaluate(model, eval_envs, logger, tag="eval")
    eval_envs.close()

    return model


if __name__ == "__main__":
    run_experiment()
