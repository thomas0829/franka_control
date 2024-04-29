import os
import joblib
import numpy as np
import hydra
import imageio
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

from utils.logger import Video, configure_logger
from utils.system import get_device, set_gpu_mode
from asid.wrapper.asid_vec import make_env, make_vec_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from asid.utils.puck import collect_rollout


@hydra.main(config_path="../configs/", config_name="task_rod_real", version_base="1.1")
def run_experiment(cfg):
    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[task][{cfg.seed}]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    set_random_seed(cfg.seed)
    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "task")
    logger = configure_logger(logdir, cfg.log.format_strings)

    cfg.robot.DoF = 2
    cfg.robot.gripper = False 
    cfg.robot.max_path_length = 1e5
    
    cfg.robot.on_screen_rendering = cfg.robot.ip_address is None

    # cfg.env.color_track = "yellow"
    cfg.env.filter = False # -> filter is buggy / inaccurate
    cfg.env.obs_keys = ["lowdim_ee", "lowdim_qpos"]
    
    cfg.asid.obs_noise = 0.0

    # Load task policy
    task_dir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "task", "policy")
    assert os.path.exists(task_dir), f"Policy not found in {task_dir}"
    policy_dict = joblib.load(task_dir)
    for k, v in policy_dict.items():
        policy_dict[k] = np.array(v)

    # Sample action
    action = np.random.normal(policy_dict["mu"], policy_dict["sigma"])
    
    # print("ALAAAARm")
    # Make environment
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid) if cfg.robot.ip_address is None else None,
        seed=cfg.seed,
        device_id=cfg.gpu_id,
        collision=False,
        verbose=False
    )

    # Load zeta parameter
    if cfg.robot.ip_address is None:
        zeta_dir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "sysid", "zeta")
        assert os.path.exists(zeta_dir), f"Zeta not found in {zeta_dir}"
        zeta_dict = joblib.load(zeta_dir)
        for k, v in zeta_dict.items():
            zeta_dict[k] = np.array(v)

        env.set_parameters(zeta_dict["real_zeta"])

    # # clip manual action and convert weight block position to center of mass
    # def normalize(value, x0, x1, y0, y1):
    #     """
    #     Normalize a value from a range [x0, x1] to a new range [y0, y1].
    #     Returns:
    #     - The normalized value.
    #     """
    #     return y0 + ((value - x0) * (y1 - y0)) / (x1 - x0)
    # action = -0.14
    # action = np.clip(action, -0.1, 0.1)
    # action = normalize(action, -0.1, 0.1, -0.07, 0.07)
    
    from asid.utils.puck import pre_reset_env_mod
    pre_reset_env_mod(env, cfg, explore=True)
    env.reset()

    # import time
    # time.sleep(5.)

    # env.reset()

    # Execute
    # action = 0.14 # 0.11
    board_x = 0.67
    # goal_x = board_x + 0.155
    goal_x = board_x + 0.305
    # goal_x = board_x + 0.455
    reward, imgs = collect_rollout(
        env, cfg, action, goal_x=None, render=True, verbose=True
    )

    imageio.mimsave(os.path.join(logger.dir, "real_task.mp4"), np.stack(imgs))

    env.reset()


if __name__ == "__main__":
    run_experiment()
