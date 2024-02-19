import argparse
import os

import imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import DictDataset
from training.policies import GaussianPolicy


def train_policy(
    policy,
    dataloader,
    epochs=10,
    lr=3e-4,
    device=None,
    exp=None,
    save_dir=None,
):
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=lr,
    )

    losses = []

    for ep in tqdm(range(epochs)):
        loss = []

        for batch in iter(dataloader):
            obs, act = batch

            obs_dict = {}
            if args.modality == "state" or args.modality == "all":
                obs_dict["state"] = torch.cat(
                    (obs["lowdim_ee"], obs["lowdim_qpos"]), dim=1
                )
            if args.modality == "images" or args.modality == "all":
                obs_dict["img"] = obs["rgb"].permute(0, 3, 1, 2)

            for k in obs_dict.keys():
                obs_dict[k] = torch.tensor(
                    obs_dict[k], dtype=torch.float32, device=device
                )

            act = torch.tensor(act, dtype=torch.float32, device=device)
            policy_loss = policy.compute_loss(obs_dict, act)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            loss.append(policy_loss.item())
        losses.append(np.mean(loss))

        if ep % 100 == 0:
            tqdm.write(f"Step {ep} | Loss {np.mean(loss)}")

            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("NLL")
            plt.savefig("loss.png")
            plt.close()

            if exp is not None and save_dir is not None:
                torch.save(
                    policy.state_dict(),
                    os.path.join(args.save_dir, args.exp, "policy.pt"),
                )

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="data")
    # hardware
    parser.add_argument("--dof", type=int, default=6, choices=[3, 4, 6])
    parser.add_argument(
        "--robot_type", type=str, default="panda", choices=["panda", "fr3"]
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        default="localhost",
        choices=[None, "localhost", "172.16.0.1"],
    )
    parser.add_argument(
        "--camera_model", type=str, default="realsense", choices=["realsense", "zed"]
    )
    # trajectories
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--modality", type=str, default="all", choices=["state", "images", "all"]
    )
    parser.add_argument("--hidden_dims", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_episode_length", type=int, default=100)

    args = parser.parse_args()

    assert args.exp is not None, "Specify --exp"

    device = torch.device(
        ("cuda:" + str(args.gpu_id))
        if args.gpu_id >= 0.0 and torch.cuda.is_available()
        else "cpu"
    )

    from os import listdir
    from os.path import isfile, join

    # get all filenames
    path = os.path.join(args.save_dir, args.exp)
    filenames = [
        f
        for f in listdir(path)
        if isfile(join(path, f))
        and f.split("_")[0] == "traj"
        and f.split(".")[-1] == "gz"
    ]

    # load single sample
    obs_keys = ["lowdim_ee", "lowdim_qpos", "207322251049_rgb"]
    obs_dict = {}
    acts = []
    obs, act = joblib.load(os.path.join(path, filenames[0]))
    for k in obs_keys:
        obs_dict[k] = [obs[k]]
    acts.append(act)

    # setup obs and act space
    state_obs_shape = None
    img_obs_shape = None
    if args.modality == "state" or args.modality == "all":
        state_obs_shape = (
            obs_dict["lowdim_ee"][0].shape[1] + obs_dict["lowdim_qpos"][0].shape[1],
        )
    if args.modality == "images" or args.modality == "all":
        img_shape = [0, 480, 480, 3]  # obs_dict["207322251049_rgb"][0].shape
        img_obs_shape = (img_shape[3], img_shape[1], img_shape[2])
    act_shape = (acts[0].shape[1],)

    policy = GaussianPolicy(
        act_shape,
        state_obs_shape=state_obs_shape,
        state_embed_dim=64,
        img_obs_shape=img_obs_shape,
        img_embed_dim=256,
        hidden_dims=[args.hidden_dims],
    ).to(device)

    if args.mode == "train":

        # load rest of the data
        for filename in tqdm(filenames[1:]):
            obs, act = joblib.load(os.path.join(path, filename))
            for k in obs_keys:
                obs_dict[k].append(obs[k])
            acts.append(act)

        for k in obs_keys:
            obs_dict[k] = np.concatenate(obs_dict[k], axis=0)
        acts = np.concatenate(acts, axis=0)

        # crop images
        obs_dict["rgb"] = obs_dict.pop("207322251049_rgb")[:, :, 50:530]

        # push in dataloader
        dataset = DictDataset(obs_dict, acts, device=device)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        policy = train_policy(
            policy,
            dataloader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            exp=args.exp,
            save_dir=args.save_dir,
        )
        torch.save(
            policy.state_dict(), os.path.join(args.save_dir, args.exp, "policy.pt")
        )

    elif args.mode == "test":
        policy.load_state_dict(
            torch.load(
                os.path.join(args.save_dir, args.exp, "policy.pt"), map_location=device
            )
        )
        policy = policy.to(device)

        from robot.robot_env import RobotEnv

        env = RobotEnv(
            control_hz=10,
            DoF=args.dof,
            robot_type=args.robot_type,
            ip_address=args.ip_address,
            camera_model=args.camera_model,
            max_lin_vel=1.0,
            max_rot_vel=1.0,
            max_path_length=args.max_episode_length,
        )

        obs = env.reset()
        assert "img_obs_0" in obs.keys(), "ERROR: camera not connected!"

        imgs = []

        for i in range(args.max_episode_length):
            imgs.append(obs["img_obs_0"])

            obs_dict = {}
            if args.modality == "state" or args.modality == "all":
                obs_dict["state"] = np.concatenate(
                    (obs["lowdim_ee"][None], obs["lowdim_qpos"][None]), axis=1
                )
            if args.modality == "images" or args.modality == "all":
                obs_dict["img"] = obs["img_obs_0"][None].transpose(0, 3, 1, 2)

            for k in obs_dict.keys():
                obs_dict[k] = torch.tensor(
                    obs_dict[k], dtype=torch.float32, device=device
                )

            act = policy(obs_dict, deterministic=False)[0].detach().cpu().numpy()

            next_obs, rew, done, _ = env.step(act)
            obs = next_obs

        # imageio.mimsave("real_rollout.mp4", np.stack(imgs), fps=30)
        imageio.mimsave("real_rollout.gif", np.stack(imgs), duration=3)
