import os
import time
import hydra
import joblib

from utils.experiment import (
    setup_wandb,
    hydra_to_dict,
    set_random_seed,
)
from utils.system import set_gpu_mode, get_device
from utils.logger import configure_logger, Video

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils.experiment import (
    setup_wandb,
    hydra_to_dict,
    set_random_seed,
)

import numpy as np
from asid.wrapper.asid_vec import make_env, make_vec_env

from utils.pointclouds import *

def viz_points(points):
    '''quick visualization: viz_points(env.get_points())'''
    points = np.concatenate((points[0,0], points[0,1]), axis=0)
    points = crop_points(points)
    visualize_pcds([points_to_pcd(points)])

@hydra.main(config_path="../configs/", config_name="explore_rod_real", version_base="1.1")
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

    cfg.num_workers = 1

    # real env
    envs = make_vec_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        # asid_cfg_dict=hydra_to_dict(cfg.asid),
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        device_id=0,
        verbose=False,
    )

    # zeta = np.random.uniform(
    #     -1.0, 1.0, (env.num_envs, env.get_parameters()[0].shape[0])
    # )
    # # set parameters, so they're not randomized on reset
    # env.set_parameters(zeta)

    # if cfg.env.sim:
    #     env.set_sphere_pos(np.array([[0.605, -0.152, 0.037]]))

    # visualize_env_points(env.get_points())
    
    ckptdir = os.path.join(
                    # logdir, "policy_step_9001" # custom ckpt
                    logdir, "policy"
                )
    
    from stable_baselines3 import SAC
    policy = SAC("MlpPolicy", envs, device=device)
    policy = policy.load(ckptdir)
    
    # if cfg.explore_exp_id is None:
    #         cfg.explore_exp_id = cfg.exp_id
            
    # if cfg.train.policy_type == "expert":
    #     cfg_env_dict["sim"] = True
    #     cfg_env_dict["ip_address"] = None
    #     policy = load_envs(cfg_env_dict, num_workers=-1, seed=cfg.seed)
    # elif cfg.train.policy_type == "rnnpolicy":
    #     from models import RNNPolicy
    #     policy = RNNPolicy(
    #         obs_size=env.observation_space.shape[0],
    #         act_size=env.action_space.shape[0],
    #         belief_size=512,
    #         hidden_size=512,
    #     ).to(device)
    #     policy.load_state_dict(
    #         torch.load(
    #             os.path.join(
    #                 cfg.log.dir, cfg.explore_exp_id, str(cfg.seed), "explore", "policy.pt"
    #             )
    #         )
    #     )
    # elif cfg.train.policy_type == "sac":
    #     from stable_baselines3 import SAC
    #     policy = SAC("MlpPolicy", env, device=device)
    #     policy = policy.load(os.path.join(
    #         cfg.log.dir, cfg.explore_exp_id, str(cfg.seed), "explore", "policy"
    #     ))
    # elif cfg.train.policy_type == "ppo":
    #     from stable_baselines3 import PPO
    #     policy = PPO("MlpPolicy", env, device=device)
    #     policy = policy.load(os.path.join(
    #                         cfg.log.dir, cfg.explore_exp_id, str(cfg.seed), "explore", "policy"
    #                     ))
        
    data = {
        "obs": [],
        "act": [],
        "rgbd": [],
    }

    # if cfg.env.sim and (hasattr(cfg.train, "ood_params") and cfg.train.ood_params):
    #     param_dim = env.get_parameters()[0].shape[0]
    #     # sample params in (normalized) range [-1.6, -1.1] or [1.1, 1.6]
    #     rnd = np.random.uniform(low=-0.5, high=0.5, size=(1, param_dim))
    #     param_ood = rnd + np.sign(rnd) * 1.1
    #     env.set_parameters(param_ood)
    
    envs.seed(cfg.seed)
    obs = envs.reset()

    # moves robot up and down to get unoccluded point clouds
    # sets _sphere_radius in robot_env which is used to crop point clouds later  
    # img_array = env.render_up()[0]
    # data["rgbd"].append(img_array)

    # if hasattr(cfg.env, "fix_zeta"):
    #     env.seed(cfg.seed)
    #     env.set_parameters(np.array(cfg.env.fix_zeta)[None])
    #     obs = env.reset()
    #     del cfg.env.fix_zeta

    # data["views"] = env.get_views()[0]
    # data["zeta"] = np.array(env.get_parameters()[0]) if cfg.env.sim else np.zeros(2)

    # if cfg.train.policy_type == "rnnpolicy":
    #     belief = policy.init_belief(1, device)

    done = False
    while not done:

        images_array = envs.render() # env.get_images_array()[0]
        data["rgbd"].append(images_array)

        # if cfg.train.policy_type == "expert":
        #     act = policy.get_demo_action(obs[0])

        # elif cfg.train.policy_type == "random":
        #     act = env.action_space.sample()
        # else:
        #     obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        #     with torch.no_grad():
        #         if cfg.train.policy_type == "sac":
        #             act, _ = policy.predict(obs, deterministic=False)
        #         elif cfg.train.policy_type == "ppo":
        #             act, _ = policy.predict(obs, deterministic=False)
        #         elif cfg.train.policy_type == "rnnpolicy":
        #             belief, action = policy.step(belief, obs_tensor, deterministic=False)
        #             act = action.cpu().numpy()

        act, _ = policy.predict(obs, deterministic=False)
        next_obs, reward, done, info = envs.step(act)

        print(f"EE {np.around(obs[0,:2],3)} Obj {np.around(obs[0,11:13],3)} Act {np.around(act,3)}")

        data["act"].append(act)
        data["obs"].append(obs)
        obs = next_obs

    
    # img_array = env.render_up()[0]
    # data["rgbd"].append(img_array)

    for k, v in data.items():
        data[k] = np.stack(v)

    import imageio
    imageio.mimwrite("explore.gif", data["rgbd"].squeeze(), duration=10)
    # b, t, c, h, w
    video = np.transpose(data["rgbd"][..., :3], (1, 0, 4, 2, 3))
    logger.record(
        f"eval_policy/traj",
        Video(video, fps=10),
        exclude=["stdout"],
    )
    logger.dump(step=0)

    # joblib.dump(data, os.path.join(logdir, f"rollout_{'sim' if cfg.env.sim else 'real'}.pkl"))
    # for rod that doesn're require reconstruction
    filenames = joblib.dump(data, os.path.join(logdir, f"rollout.pkl"))
    print("Saved rollout to", filenames)
    # print(os.path.join(logdir, f"rollout_{'sim' if cfg.env.sim else 'real'}.pkl"))

if __name__ == "__main__":
    run_experiment()
