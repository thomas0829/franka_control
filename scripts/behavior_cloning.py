import os
import joblib
import imageio
import argparse
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt

from training.policies import GaussianPolicy

from torch.utils.data import DataLoader
from training.dataset import DictDataset


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
                obs_dict["img"] = obs["img_obs_0"].permute(0, 3, 1, 2)

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
<<<<<<< HEAD
    parser.add_argument("--robot_type", type=str, default="panda", choices=["panda", "fr3"])
    parser.add_argument("--ip_address", type=str, default="localhost", choices=[None, "localhost", "172.16.0.1"])
    parser.add_argument("--camera_model", type=str, default="realsense", choices=["realsense", "zed"])
=======
    parser.add_argument(
        "--robot_type", type=str, default="panda", choices=["panda", "fr3"]
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        default="localhost",
        choices=[None, "localhost", "172.16.0.1"],
    )
    parser.add_argument("--camera_ids", type=list, default=[])
    parser.add_argument(
        "--camera_model", type=str, default="realsense", choices=["realsense", "zed"]
    )
>>>>>>> e8b764b87a1a33b3459e555d29be9acfd137157f
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

<<<<<<< HEAD
    assert args.exp is not None, "Specify --exp"

    device = torch.device(("cuda:" + str(args.gpu_id)) if args.gpu_id >= 0. and torch.cuda.is_available() else "cpu")
=======
    args.exp = "pick_red"
    device = torch.device(("cuda:" + str(args.gpu_id)) if args.gpu_id >= 0. else "cpu")
>>>>>>> e8b764b87a1a33b3459e555d29be9acfd137157f

    buffer = joblib.load(os.path.join(args.save_dir, args.exp, "buffer.gz"))
    for k in buffer.observations.keys():
        buffer.observations[k] = buffer.observations[k][: len(buffer)]
    buffer.actions = buffer.actions[: len(buffer)]

<<<<<<< HEAD
=======
    dataset = DictDataset(buffer.observations, buffer.actions, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

>>>>>>> e8b764b87a1a33b3459e555d29be9acfd137157f
    # idcs = joblib.load(os.path.join(args.save_dir, args.exp, "idcs.gz"))
    
    # imgs = buffer.observations["img_obs_0"]
    # imageio.mimsave("real_rollout.mp4", imgs.cpu().numpy(), fps=30)
    # print(f"Loaded {len(buffer)} time steps")

    state_obs_shape = None
    img_obs_shape = None
    if args.modality == "state" or args.modality == "all":
        state_obs_shape = (
            buffer.observations["lowdim_ee"].shape[1]
            + buffer.observations["lowdim_qpos"].shape[1],
        ) 
    if args.modality == "images" or args.modality == "all":
        img_shape = buffer.observations["img_obs_0"].shape
        img_obs_shape = (img_shape[3], img_shape[1], img_shape[2])

    act_shape = (buffer.actions.shape[1],)
    policy = GaussianPolicy(
        act_shape,
        state_obs_shape=state_obs_shape,
        state_embed_dim=64,
        img_obs_shape=img_obs_shape,
        img_embed_dim=256,
        hidden_dims=[args.hidden_dims],
    ).to(device)

    if args.mode == "train":
<<<<<<< HEAD

        dataset = DictDataset(buffer.observations, buffer.actions, device=device)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

=======
>>>>>>> e8b764b87a1a33b3459e555d29be9acfd137157f
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
<<<<<<< HEAD
        policy.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp, "policy.pt"), map_location=device))
=======
        policy.load_state_dict(
            torch.load(os.path.join(args.save_dir, args.exp, "policy.pt"))
        )
>>>>>>> e8b764b87a1a33b3459e555d29be9acfd137157f
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
<<<<<<< HEAD
        assert "img_obs_0" in obs.keys(), "ERROR: camera not connected!"

=======
>>>>>>> e8b764b87a1a33b3459e555d29be9acfd137157f
        imgs = []

        for i in range(args.max_episode_length):
            imgs.append(obs["img_obs_0"])

            obs_dict = {}
            if args.modality == "state" or args.modality == "all":
                obs_dict["state"] = np.concatenate(
<<<<<<< HEAD
                    (obs["lowdim_ee"][None], obs["lowdim_qpos"][None]), axis=1
                )
            if args.modality == "images" or args.modality == "all":
                obs_dict["img"] = obs["img_obs_0"][None].transpose(0, 3, 1, 2)
=======
                    (obs["lowdim_ee"], obs["lowdim_qpos"]), axis=1
                )
            if args.modality == "images" or args.modality == "all":
                obs_dict["img"] = obs["img_obs_0"].transpose(0, 3, 1, 2)
>>>>>>> e8b764b87a1a33b3459e555d29be9acfd137157f

            for k in obs_dict.keys():
                obs_dict[k] = torch.tensor(
                    obs_dict[k], dtype=torch.float32, device=device
                )

<<<<<<< HEAD
            act = policy(obs_dict, deterministic=False)[0].detach().cpu().numpy()
=======
            act = policy(obs, deterministic=False)[0].cpu().detach().numpy()
>>>>>>> e8b764b87a1a33b3459e555d29be9acfd137157f

            next_obs, rew, done, _ = env.step(act)
            obs = next_obs

<<<<<<< HEAD
        # imageio.mimsave("real_rollout.mp4", np.stack(imgs), fps=30)
        imageio.mimsave("real_rollout.gif", np.stack(imgs), duration=3)
=======
        imageio.mimsave("real_rollout.gif", np.stack(imgs), duration=0.5)
>>>>>>> e8b764b87a1a33b3459e555d29be9acfd137157f
