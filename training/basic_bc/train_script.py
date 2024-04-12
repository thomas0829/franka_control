import io
import os
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.basic_bc.datasets.state_image_dataset import (
    HDF5StateImageDataset, StateImageDataset)
from training.basic_bc.models.policies import MixedGaussianPolicy
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Image, Video, configure_logger
from utils.system import get_device, set_gpu_mode


def plot_trajectory(pred_actions, true_actions=None, imgs=None):
    # expecting imgs T, N, C, H, W where N is the number of images

    # https://github.com/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb
    ACTION_DIM_LABELS = ["x", "y", "z", "yaw", "pitch", "roll", "grasp"]

    # build image strip to show above actions
    if imgs is not None:
        img_strip = np.concatenate(imgs.squeeze()[::6].transpose(0, 2, 3, 1), axis=1)

    # set up plt figure
    figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    # plot actions
    pred_actions = np.array(pred_actions).squeeze()
    if true_actions is not None:
        true_actions = np.array(true_actions).squeeze()
    for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
        # actions have batch, horizon, dim, in this example we just take the first action for simplicity
        axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
        if true_actions is not None:
            axs[action_label].plot(true_actions[:, action_dim], label="ground truth")
        axs[action_label].set_title(action_label)
        axs[action_label].set_xlabel("Time in one episode")

    if imgs is not None:
        axs["image"].imshow(img_strip)
    axs["image"].set_xlabel("Time in one episode (subsampled)")
    plt.legend()
    plt.tight_layout()

    # figure to buffer to PIL to numpy
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = PILImage.open(buf)
    image_array = np.array(image)
    buf.close()
    plt.close(fig)

    return image_array


def evaluate(policy, dataloader, cfg, device, n_traj=5):

    episode_ends = dataloader.dataset.episode_ends[:n_traj]
    episode_ends = np.concatenate([np.array([0]), episode_ends])
    plot_imgs = []

    for i in range(len(episode_ends) - 1):

        idx_start = np.sum(episode_ends[: i + 1])
        idx_end = np.sum(episode_ends[: i + 2])

        states = None
        if len(cfg.training.state_keys):
            states = dataloader.dataset.normalized_train_data["state"][
                idx_start:idx_end
            ]
        imgs = None
        if len(cfg.training.image_keys):
            imgs = dataloader.dataset.normalized_train_data["image"][idx_start:idx_end]
        true_actions = dataloader.dataset.normalized_train_data["action"][
            idx_start:idx_end
        ]

        with torch.no_grad():
            imgs_torch = torch.tensor(imgs).to(device) if imgs is not None else None
            states_torch = (
                torch.tensor(states).to(device) if states is not None else None
            )
            acts = policy.forward(
                imgs=imgs_torch, states=states_torch, deterministic=False
            )

            mse = nn.functional.mse_loss(
                acts, torch.tensor(true_actions).to(device)
            ).item()

            pred_actions = acts.cpu().numpy()

        plot_imgs.append(
            plot_trajectory(pred_actions, true_actions=true_actions, imgs=imgs[:, 0])
        )

    return mse, plot_imgs


@hydra.main(version_base=None, config_path="../../configs", config_name="bc_policy_sim")
def run_experiment(cfg):

    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[{cfg.seed}][train]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    set_random_seed(cfg.seed)

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed))
    logger = configure_logger(logdir, cfg.log.format_strings)

    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    if "helen" in cfg.training.dataset_path:
        HDF5StateImageDataset
        train_dataset = HDF5StateImageDataset(
            os.path.join(cfg.training.dataset_path, "train"),
            num_trajectories=cfg.training.num_trajectories,
            image_keys=cfg.training.image_keys,
            state_keys=cfg.training.state_keys,
        )
        train_stats = train_dataset.stats

        eval_dataset = None

    else:
        train_dataset = StateImageDataset(
            os.path.join(cfg.training.dataset_path, "train"),
            num_trajectories=cfg.training.num_trajectories,
            image_keys=cfg.training.image_keys,
            state_keys=cfg.training.state_keys,
        )
        train_stats = train_dataset.stats

        eval_dataset = StateImageDataset(
            os.path.join(cfg.training.dataset_path, "eval"),
            num_trajectories=cfg.training.num_trajectories,
            image_keys=cfg.training.image_keys,
            state_keys=cfg.training.state_keys,
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=torch.Generator("cuda"),
    )

    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            generator=torch.Generator("cuda"),
        )

    policy = MixedGaussianPolicy(
        img_shape=(
            train_stats["image"]["max"].shape if len(cfg.training.image_keys) else None
        ),
        state_shape=(
            train_stats["state"]["max"].shape if len(cfg.training.state_keys) else None
        ),
        act_shape=train_stats["action"]["max"].shape,
        hidden_dim=cfg.training.hidden_dim,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    min_loss = np.inf
    with tqdm(
        range(cfg.training.num_epochs),
        desc="Epoch",
    ) as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            epoch_mse = list()
            epoch_grad_norm = list()
            # batch loop
            with tqdm(train_dataloader, desc="Batch", leave=False) as tepoch:
                for batch in tepoch:

                    states = None
                    if len(cfg.training.state_keys):
                        states = batch["state"].to(device)
                    imgs = None
                    if len(cfg.training.image_keys):
                        imgs = batch["image"].float().to(device)
                    acts = batch["action"].to(device)

                    # forward pass
                    loss, dist = policy.compute_loss(acts, imgs, states)

                    # backward pass
                    loss.backward()

                    # compute metrics
                    with torch.no_grad():
                        mse_cpu = nn.functional.mse_loss(dist.mean, acts).item()
                    grad_norm_cpu = (
                        torch.cat(
                            [
                                p.grad.flatten()
                                for p in policy.parameters()
                                if p.grad is not None
                            ]
                        )
                        .norm()
                        .item()
                    )

                    # clip gradients
                    if cfg.training.clip_grad_norm > 0:
                        nn.utils.clip_grad_norm_(
                            policy.parameters(), cfg.training.clip_grad_norm
                        )

                    # optimize
                    optimizer.step()
                    optimizer.zero_grad()

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    epoch_mse.append(mse_cpu)
                    epoch_grad_norm.append(grad_norm_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # Log metrics
            logger.record("nll_loss", np.mean(epoch_loss))
            logger.record("mse_loss", np.mean(epoch_mse))
            logger.record("learning_rate", optimizer.param_groups[0]["lr"])
            logger.record("grad_norm (unclipped)", np.mean(epoch_grad_norm))
            logger.record("policy_mean (batch)", torch.mean(dist.mean).item())
            logger.record("policy_std (batch)", torch.mean(dist.stddev).item())

            # Visualize policy
            if epoch_idx % cfg.training.eval_interval == 0:
                n_traj = 10
                train_mse, plot_imgs = evaluate(
                    policy, train_dataloader, cfg, device, n_traj=n_traj
                )
                if plot_imgs is not None:
                    for i, plot_img in enumerate(plot_imgs):
                        logger.record(
                            f"train/trajectory_{i}",
                            Image(plot_img, dataformats="HWC"),
                            exclude=["stdout"],
                        )
                logger.record(f"train/mse_loss ({n_traj} samples)", train_mse)

                if eval_dataset is not None:
                    eval_mse, plot_imgs = evaluate(
                        policy, eval_dataloader, cfg, device, n_traj=n_traj
                    )
                    if plot_imgs is not None:
                        for i, plot_img in enumerate(plot_imgs):
                            logger.record(
                                f"eval/trajectory_{i}",
                                Image(plot_img, dataformats="HWC"),
                                exclude=["stdout"],
                            )
                    logger.record(f"eval/mse_loss ({n_traj} samples)", eval_mse)

            # Dump logs
            logger.dump(step=epoch_idx)

            # Save checkpoint
            checkpoint = {
                "state_dict": policy.state_dict(),
                "stats": train_stats,
                "config": cfg,
            }
            # Always save the best model
            if np.mean(epoch_loss) < min_loss:
                min_loss = np.mean(epoch_loss)
                torch.save(checkpoint, os.path.join(logdir, f"bc_policy"))
            # Sometimes save the ckpt model
            if epoch_idx % cfg.training.save_interval == 0:
                torch.save(checkpoint, os.path.join(logdir, f"bc_policy_{epoch_idx}"))


if __name__ == "__main__":
    run_experiment()
