import os
import joblib
import imageio
import argparse
from tqdm import trange

import torch
import numpy as np
import matplotlib.pyplot as plt

from training.policies import GaussianPolicy

def train_policy(policy, buffer, epochs=10, batch_size=16, lr=3e-4, device=None):
    
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=lr,
    )

    losses = []

    for step in trange(epochs):
        
        obs, act, rew, next_obs, done = buffer.sample(batch_size=batch_size)

        obs_dict = {}
        if args.modality == "state" or args.modality == "all":
            obs_dict["state"] = np.concatenate((obs["lowdim_ee"], obs["lowdim_qpos"]), axis=1)
        if args.modality == "images" or args.modality == "all":
            obs_dict["img"] = obs["img_obs_0"].transpose(0,3,1,2)
        
        for k in obs_dict.keys():
            obs_dict[k] = torch.tensor(obs_dict[k], dtype=torch.float32, device=device)

        act = torch.tensor(act, dtype=torch.float32, device=device)
        policy_loss = policy.compute_loss(obs_dict, act)
        
        losses.append(np.clip(policy_loss.item(),-10,10))
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if step % 100 == 0:
            plt.plot(losses)
            plt.savefig("loss.png")
            plt.close()

    return policy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="data")
    # hardware
    parser.add_argument("--dof", type=int, default=6, choices=[3, 4, 6])
    parser.add_argument("--robot_type", type=str, default="panda", choices=["panda", "fr3"])
    parser.add_argument("--ip_address", type=str, default="localhost", choices=[None, "localhost", "172.16.0.1"])
    parser.add_argument("--camera_ids", type=list, default=[])
    parser.add_argument("--camera_model", type=str, default="realsense", choices=["realsense", "zed"])
    # trajectories
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--modality", type=str, default="state", choices=["state", "images", "all"])
    parser.add_argument("--hidden_dims", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_episode_length", type=int, default=100)

    args = parser.parse_args()

    buffer = joblib.load(os.path.join(args.save_dir, args.exp, "buffer.gz"))
    
    state_obs_shape = None
    img_obs_shape = None
    if args.modality == "state" or args.modality == "all":
        state_obs_shape = (buffer.observations["lowdim_ee"].shape[1] + buffer.observations["lowdim_qpos"].shape[1], )    
    if args.modality == "images" or args.modality == "all":
        img_shape = buffer.observations["img_obs_0"].shape
        img_obs_shape = (img_shape[3], img_shape[1], img_shape[2])
   
    act_shape = (buffer.actions.shape[1], )
    policy = GaussianPolicy(act_shape, state_obs_shape=state_obs_shape, state_embed_dim=32, img_obs_shape=img_obs_shape, img_embed_dim=64, hidden_dims=[args.hidden_dims])

    device = torch.device(("cuda:" + str(args.gpu_id)) if args.gpu_id else "cpu")
    
    if args.mode == "train":
        policy = train_policy(policy, buffer, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
        torch.save(policy.state_dict(), os.path.join(args.save_dir, args.exp, "policy.pt"))

    elif args.mode == "test":
        policy.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp, "policy.pt")))
        policy = policy.to(device)

        from robot.robot_env import RobotEnv

        env = RobotEnv(
            control_hz=10,
            DoF=args.dof,
            robot_type=args.robot_type,
            ip_address=args.ip_address,
            camera_ids=args.camera_ids,
            camera_model=args.camera_model,
            max_lin_vel=1.0,
            max_rot_vel=1.0,
            max_path_length=args.max_episode_length,
        )

        obs = env.reset()
        
        for i in range(args.max_episode_length):
            
            obs_dict = {}
            if args.modality == "state" or args.modality == "all":
                obs_dict["state"] = np.concatenate((obs["lowdim_ee"], obs["lowdim_qpos"]), axis=1)
            if args.modality == "images" or args.modality == "all":
                obs_dict["img"] = obs["img_obs_0"].transpose(0,3,1,2)

            for k in obs_dict.keys():
                obs_dict[k] = torch.tensor(obs_dict[k], dtype=torch.float32, device=device)

            act = policy(obs, deterministic=False)[0].cpu().detach().numpy()
            
            next_obs, rew, done, _ = env.step(act)
            obs = next_obs
            

        # TODO save gif w/ imageio