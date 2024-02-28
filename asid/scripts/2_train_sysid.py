import os
import hydra
import numpy as np
import joblib
import imageio

from asid.wrapper.asid_vec import make_env, make_vec_env
from asid.sysid.identifier import SysIdentifier
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Video, configure_logger
from utils.system import get_device, set_gpu_mode


@hydra.main(config_path="../configs/", config_name="sysid_rod_sim", version_base="1.1")
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

    # train env
    envs_sysid = make_vec_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        device_id=0,
    )

    rollout = joblib.load(os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "explore", "rollout.pkl"))
    
    # # convert angles to mujoco quaternion and
    # set obj pose
    obj_pose = rollout["obs"][0,...,-7:].copy()
    # obj_pose[:,3:] = euler_to_quat_mujoco(quat_to_euler(obj_pose[:,3:]))
    obj_pose = np.repeat(obj_pose, cfg.num_workers, axis=0)
    envs_sysid.set_obj_pose(obj_pose)

    # # convert angles to mujoco quaternion
    # for i in range(rollout["obs"].shape[0]):
    #     rollout["obs"][i,...,-4:] = euler_to_quat_mujoco(quat_to_euler(rollout["obs"][i,...,-4:]))

    params_dim = envs_sysid.get_parameters()[0].shape[0]
    
    imageio.mimsave(f"real.gif", np.stack(rollout["rgbd"][..., :3])[:,0], duration=3)
    video = np.transpose(rollout["rgbd"][..., :3], (1, 0, 4, 2, 3))
    logger.record(
        f"real/trajectory",
        Video(video, fps=20),
        exclude=["stdout"],
    )

    Js = []
    means = []
    stds = []

    for i in range(cfg.train.n_restarts):

        # setup identifier
        identifier = SysIdentifier(
            # env
            zeta_dim=params_dim,
            num_workers=cfg.num_workers,
            # bbo
            **cfg.train.algorithm,
            **cfg.train.distribution,
            # train
            n_epochs=cfg.train.n_epochs,
            fit_per_epoch=cfg.train.fit_per_epoch,
            population_size=cfg.train.population_size,
            seed=cfg.seed,
            rnd_env_seed=cfg.train.rnd_env_seed,
            # log
            logger=logger,
            save_interval=cfg.log.save_interval,
            eval_interval=cfg.log.eval_interval,
            verbose=True,
        )

        # sphere pos, real action sequence
        rollout["zeta"] = np.zeros(params_dim)
        mean_fit, mean, std = identifier.identify(
            rollout["obs"], rollout["act"], rollout["zeta"], envs_sysid
        )

        Js.append(mean_fit)
        means.append(mean)
        stds.append(std)

        print("restart", i, mean_fit, mean, std)

    best_idx = np.argmax(Js)
    
    logger.record(
            f"result/J",
            Js[best_idx]
        )
    for i, mean in enumerate(means[best_idx]):
        logger.record(
            f"result/mean_{i}",
            mean
        )

    param_dict = {
        "real_zeta": rollout["zeta"],
        "mu": means[best_idx],
        "sigma": stds[best_idx],
        "final_obs": rollout["obs"][-1]
    }
    joblib.dump(param_dict, os.path.join(identifier.logger.dir, "zeta"))

    logger.dump(step=0)

if __name__ == "__main__":
    run_experiment()
