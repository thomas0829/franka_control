import json
import os

import hydra
import matplotlib.pyplot as plt
from nvisii_renderer import NVISIIRenderer

from asid.wrapper.asid_vec import make_env
from utils.experiment import hydra_to_dict
from utils.transformations import *
from utils.transformations_mujoco import *


@hydra.main(config_path="../asid/configs/", config_name="explore_rod_sim", version_base="1.1")
def run_experiment(cfg):

    cfg.robot.DoF = 6
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )
    env.reset()

    with open(os.path.join("rendering", "nvisii_config.json")) as f:
        renderer_config = json.load(f)

    viewer = NVISIIRenderer(env=env.unwrapped._robot, **renderer_config)
    viewer.reset()

    camera_extrinsic = env.unwrapped._robot.get_camera_extrinsic("xx")
    pos = camera_extrinsic[:3,3:]
    quat = mat_to_quat_mujoco(camera_extrinsic[:3,:3])

    # pos = [1.3420383158842424, -0.037408864541261, 0.5281817817238226]
    # quat= euler_to_quat(quat_to_euler_mujoco(np.array([0.60670078, 0.37513562, 0.37637644, 0.5912091])))
    quat = rmat_to_quat(camera_extrinsic[:3,:3])
    # quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    pos = [1.1, 0.5, 0.8]
    viewer.set_camera_pos_quat(pos, camera_extrinsic[:3,:3])

    for i in range(5):
        viewer.update()

        env.step(env.action_space.sample())
        viewer.render()
        
    viewer.close()


if __name__ == "__main__":
    run_experiment()
