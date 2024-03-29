import os
import time

import hydra
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Video, configure_logger
from utils.system import get_device, set_gpu_mode

from training.basic_bc.models.policies import MixedGaussianPolicy
from training.basic_bc.datasets.state_image_dataset import StateImageDataset


@hydra.main(
    version_base=None, config_path="../../configs", config_name="bc_policy_real"
)
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

    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    dataset = StateImageDataset(
        cfg.training.dataset_path,
        num_trajectories=cfg.training.num_trajectories,
        image_keys=cfg.training.image_keys,
        state_keys=cfg.training.state_keys,
    )
    stats = dataset.stats

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=torch.Generator("cuda"),
    )

    policy = MixedGaussianPolicy(
        img_shape=stats["image"]["max"].shape[1:] if len(cfg.training.image_keys) else None,
        state_shape=stats["state"]["max"].shape if len(cfg.training.state_keys) else None,
        act_shape=stats["action"]["max"].shape,
        hidden_dim=128,
    ).to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=3e-4,
        weight_decay=1e-2,
    )

    with tqdm(
        range(cfg.training.num_epochs),
        desc="Epoch",
    ) as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for batch in tepoch:
                    # data normalized in dataset
                    # device transfer

                    imgs = None
                    states = None
                    acts = batch["action"].to(device)

                    # forward pass
                    if len(cfg.training.state_keys):
                        states = batch["state"].to(device)
                    if len(cfg.training.image_keys):
                        imgs = batch["image"].to(device)
                    loss = policy.compute_loss(acts, imgs, states)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            logger.record("nll_loss", np.mean(epoch_loss))
            logger.record("learning_rate", optimizer.param_groups[0]["lr"])

            logger.dump(step=epoch_idx)

            # Save checkpoint
            # Create a state dict with the paramaters of the policy, and the stats of the dataset
            if epoch_idx % 10 == 0:
                checkpoint = {
                    "state_dict": policy.state_dict(),
                    "stats": stats,
                    "config": cfg,
                }
                torch.save(checkpoint, os.path.join(logdir, f"bc_policy_{epoch_idx}"))

    # Save checkpoint
    # Create a state dict with the paramaters of the policy, and the stats of the dataset
    checkpoint = {"state_dict": policy.state_dict(), "stats": stats, "config": cfg}
    torch.save(checkpoint, os.path.join(logdir, "bc_policy"))


if __name__ == "__main__":
    run_experiment()
