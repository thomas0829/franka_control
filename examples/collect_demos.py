import os
import time
import numpy as np
from tqdm import tqdm
import hydra

from robot.controllers.oculus import VRController
from robot.rlds_wrapper import (
    convert_rlds_to_np,
    load_rlds_dataset,
    wrap_env_in_rlds_logger,
)
from robot.robot_env import RobotEnv
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict


@hydra.main(
    config_path="../configs/", config_name="collect_demos_real", version_base="1.1"
)
def run_experiment(cfg):

    data_dir = os.path.join(cfg.data_dir, cfg.exp_id, str(cfg.seed))
    os.makedirs(data_dir, exist_ok=True)

    cfg.robot.max_path_length = cfg.max_episode_length

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    oculus = VRController()
    assert oculus.get_info()["controller_on"], "ERROR: oculus controller off"
    print("Oculus Connected")

    # assert env._num_cameras > 0, "ERROR: camera(s) not connected!"

    with wrap_env_in_rlds_logger(
        env, cfg.exp_id, data_dir, max_episodes_per_shard=1 # cfg.episodes
    ) as rlds_env:
        for i in range(cfg.episodes):
            # reset w/o recording obs
            rlds_env.unwrapped.reset()
            print(f"Camera(s) Connected ({rlds_env.unwrapped._num_cameras})")

            print(f"Press 'A' to Start Collecting")
            # time to reset the scene
            while True:
                info = oculus.get_info()
                if info["success"]:
                    # reset w/ recording obs after resetting the scene
                    obs = rlds_env.reset()
                    break

            print(f"Press 'B' to Stop Collecting")

            obss = []
            acts = []

            for j in tqdm(
                range(cfg.max_episode_length), desc=f"Collecting Trajectory {i}"
            ):
                # wait for controller input
                info = oculus.get_info()
                while not (info["failure"] or info["movement_enabled"]):
                    info = oculus.get_info()

                # press 'B' to end a trajectory
                if info["failure"]:
                    continue

                # check if 'trigger' button is pressed
                if info["movement_enabled"]:
                    # prepare obs for oculus
                    pose = rlds_env.unwrapped._robot.get_ee_pose()
                    gripper = rlds_env.unwrapped._robot.get_gripper_position()
                    state = {
                        "robot_state": {
                            "cartesian_position": pose,
                            "gripper_position": gripper,
                        }
                    }
                    vel_act = oculus.forward(state)

                    # convert vel to delta actions
                    delta_act = rlds_env.unwrapped._robot._ik_solver.cartesian_velocity_to_delta(
                        vel_act
                    )
                    delta_gripper = (
                        rlds_env.unwrapped._robot._ik_solver.gripper_velocity_to_delta(
                            vel_act[-1:]
                        )
                    )
                    # prepare act
                    if cfg.robot.DoF == 3:
                        act = np.concatenate((delta_act[:3], vel_act[-1:]))
                    elif cfg.robot.DoF == 4:
                        act = np.concatenate(
                            (delta_act[:3], delta_act[5:6], vel_act[-1:])
                        )
                    elif cfg.robot.DoF == 6:
                        act = np.concatenate((delta_act, vel_act[-1:]))

                    next_obs, rew, done, _ = rlds_env.step(rlds_env.type_action(act))
                    if cfg.robot.ip_address == None:
                        time.sleep(1/cfg.robot.control_hz)
                        env.render()

                    obss.append(obs)
                    acts.append(act)

                    obs = next_obs

            print(f"Recorded Trajectory {i}")

    env.reset()

    # check if dataset was saved
    loaded_dataset = load_rlds_dataset(data_dir)

    print(f"Finished Collecting {i} Trajectories")


if __name__ == "__main__":
    run_experiment()
