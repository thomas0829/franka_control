import time
import hydra

from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict


@hydra.main(config_path="../configs/", config_name="default", version_base="1.1")
def run_experiment(cfg):

    cfg.robot.DoF = 6
    cfg.robot.on_screen_rendering = False
    cfg.robot.gripper = True

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    start = time.time()
    for i in range(15):
        env.reset()
        for i in range(5):
            obs, reward, done, info = env.step(env.action_space.sample())
            env.render()
    print(time.time() - start)


if __name__ == "__main__":
    run_experiment()
