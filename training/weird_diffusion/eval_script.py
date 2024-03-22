import collections
import os
from dataclasses import dataclass

import gym
import hydra
import imageio
import numpy as np
import torch
from tqdm.auto import tqdm

from robot.sim.vec_env.vec_env import make_env
from training.weird_diffusion.datasets.utils import (normalize_data,
                                                     unnormalize_data)
from training.weird_diffusion.models.make_networks import \
    instantiate_model_artifacts
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Video, configure_logger
from utils.system import get_device, set_gpu_mode


def process_obs(obs, nets, cfg, device):
    # This function processes the observation such that they can just be fed into the model.
    # It should return a dictionary with the following keys
    # 'embed': The image embeddings
    # 'state': The state of the environment
    # You can change how you get this information depending on the environment.

    processed_obs = {}

    if len(cfg.training.state_keys) and cfg.training.with_state:
        sts = []
        for key in cfg.training.state_keys:
            sts.append(obs[key])
        state = np.concatenate(sts, axis=0)
        processed_obs['state'] = state

    if len(cfg.training.image_keys):
        with torch.no_grad():
            imgs = []
            for key in cfg.training.image_keys:
                # TODO deal with cropping properly
                img = obs[key][:, 160:, :]
                img = torch.tensor(obs[key], dtype=torch.float32).permute(2, 0, 1).to(device)
                imgs.append(img)
            im_stack = torch.stack(imgs, dim=0)
            images = nets['vision_encoder'](im_stack).cpu().flatten().numpy()
            processed_obs['embed'] = images
    
    return processed_obs

def run_one_eval(env: gym.Env, nets: torch.nn.Module, config, stats, noise_scheduler, device,
                 max_steps: int, render: bool) -> bool:
    # get first observation
    obs = env.reset()
    obs = process_obs(obs, nets, config, device)

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * config.training.obs_horizon, maxlen=config.training.obs_horizon)
    # save visualization and rewards
    rewards = list()
    imgs = list()
    done = False
    step_idx = 0

    while not done:
        B = 1
        # stack the last obs_horizon number of observations
        images = np.stack([x['embed'] for x in obs_deque])
        if config.training.with_state:
            agent_poses = np.stack([x['state'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['state'])
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

        # images are already normalized to [0,1]
        nimages = images

        # device transfer
        nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
        # (2,3,96,96)
        # (2,2)

        # infer action
        with torch.no_grad():
            # get image features
            image_features = nimages
            # (2,1024)

            # concat with low-dim observations
            if config.training.with_state:
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)
            else:
                obs_features = image_features

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Gaussian noise
            noisy_action = torch.randn(
                (B, config.training.pred_horizon, config.training.action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(config.training.num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        # only take action_horizon number of actions
        start = config.training.obs_horizon - 1
        end = start + config.training.action_horizon
        action = action_pred[start:end, :]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, done, info = env.step(action[i])
            # save observations
            obs = process_obs(obs, nets, config, device)
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            if render:
                imgs.append(env.unwrapped.render(sn=config.training.image_keys[0].split("_")[0]))

            # update progress bar
            step_idx += 1
            if step_idx > max_steps:
                return False, imgs
            if done:
                if reward > 0:
                    return True, imgs
                return False, imgs


@hydra.main(version_base=None, config_path="../../configs", config_name="diffusion_policy_sim")
def run_experiment(cfg):
    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[{cfg.seed}]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    set_random_seed(cfg.seed)
    
    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed))
    logger = configure_logger(logdir, cfg.log.format_strings)

    checkpoint = torch.load(os.path.join(logdir, "diffusion_policy"), map_location='cuda')

    nets, noise_scheduler, device = instantiate_model_artifacts(cfg, model_only=True)
    nets.load_state_dict(checkpoint['state_dict'])
    print('Pretrained weights loaded.')
    stats = checkpoint['stats']

    cfg.robot.max_path_length = cfg.inference.max_steps
    cfg.robot.blocking_control = True

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    successes = 0
    render = True
    for i in tqdm(range(cfg.inference.num_eval_episodes), desc='Evaluating'):
        succeeded, imgs = run_one_eval(env=env, nets=nets, config=cfg, stats=stats, noise_scheduler=noise_scheduler,
                     device=device, max_steps=cfg.inference.max_steps, render=render)
        if succeeded:
            successes += 1

        if render:
            video = np.stack(imgs)[None]
            imageio.mimsave(os.path.join(logdir, f"eval_episode_{i}.gif"), video[0])
            logger.record(
                f"videos/eval_episode_{i}",
                Video(video, fps=20),
                exclude=["stdout"],
            )
            
    # Round to the 3rd decimal place
    success_rate = round(successes / cfg.inference.num_eval_episodes, 3)
    print(f'Success rate: {success_rate}')


if __name__ == "__main__":
    run_experiment()