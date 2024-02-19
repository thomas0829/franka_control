import os
import imageio
import numpy as np
import hydra

from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Video, configure_logger
from utils.system import get_device, set_gpu_mode


@hydra.main(
    config_path="../configs/", config_name="default", version_base="1.1"
)
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

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    obs = env.reset()

    imgs = []
    done = False
    while not done:
        next_obs, rew, done, _ = env.step(env.action_space.sample())
        imgs.append(env.render())
        obs = next_obs

    imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)


if __name__ == "__main__":
    run_experiment()
