import os
import time
import joblib
import hydra
import numpy as np
from datetime import datetime

from asid.wrapper.asid_vec import make_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import configure_logger
from utils.system import get_device, set_gpu_mode
from utils.transformations import quat_to_euler
from utils.transformations_mujoco import euler_to_quat_mujoco

@hydra.main(
    config_path="../configs/", config_name="explore_rod_sim", version_base="1.1"
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
    set_gpu_mode(cfg.gpu_id >= 0, gpu_id=cfg.gpu_id)
    device = get_device()

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed))
    logger = configure_logger(logdir, cfg.log.format_strings)

    act_seq = np.array(
        [
            [0.1, 0.0],
            [0.1, 0.0],
            [0.1, 0.0],
            [0.1, 0.0],
            [0.1, 0.0],
            [0.1, 0.0],
            [0.1, 0.0],
            [0.0, -0.1],
            [0.0, -0.1],
            [0.1, 0.0],
            [0.1, 0.0],
        ]
    )

    # collect trajectory in REAL

    cfg.robot.ip_address = "172.16.0.1"
    cfg.robot.camera_model = "realsense"
    cfg.robot.imgs = True
    
    cfg.env.obs_keys = None
    cfg.env.flatten = False

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
        verbose=False,
    )

    obs = env.reset()
    
    obj_init = env.get_obj_pose().copy()
    obj_init[3:] = euler_to_quat_mujoco(quat_to_euler(obj_init[3:]))

    dict_real = {}
    for k in obs.keys():
        dict_real[k] = []

    for act in act_seq:
        for k in dict_real.keys():
            dict_real[k].append(obs[k])
        next_obs, rew, done, _ = env.step(act)
        obs = next_obs

    for k in dict_real.keys():
        dict_real[k] = np.stack(dict_real[k], axis=0)

    env.reset()

    # collect trajectory in SIM

    cfg.robot.ip_address = None
    cfg.robot.camera_model = None

    cfg.asid.reward = False

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    env.set_obj_pose(obj_init)
    obs = env.reset()

    dict_sim = {}
    for k in obs.keys():
        dict_sim[k] = []

    for act in act_seq:
        for k in dict_sim.keys():
            dict_sim[k].append(obs[k])
        next_obs, rew, done, _ = env.step(act)
        obs = next_obs

    for k in dict_sim.keys():
        dict_sim[k] = np.stack(dict_sim[k], axis=0)

    # dump data
    data_dict = {"real": dict_real, "sim": dict_sim}
    current_time_date = datetime.now().strftime("%y_%m_%d_%H_%M_%S")

    joblib.dump(data_dict, os.path.join(logdir, f"sim_vs_real_{current_time_date}"))


if __name__ == "__main__":
    run_experiment()
