import hydra
import numpy as np
from robot.controllers.oculus import VRController
from robot.wrappers.data_wrapper import DataCollectionWrapper
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict
from absl import flags
FLAGS = flags.FLAGS
import sys
from datetime import date

# Setup logger
import logging
from datetime import date

# Setup logger
logger = logging.getLogger("collect_demos")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

class LogColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log_config(msg):
    logger.info(f"{LogColors.CYAN}{msg}{LogColors.END}")

def log_connect(msg):
    logger.info(f"{LogColors.BLUE}{msg}{LogColors.END}")

def log_instruction(msg):
    logger.info(f"{LogColors.YELLOW}{msg}{LogColors.END}")

def log_success(msg):
    logger.info(f"{LogColors.GREEN}{msg}{LogColors.END}")

def log_failure(msg):
    logger.info(f"{LogColors.RED}{msg}{LogColors.END}")

def log_important(msg):
    logger.info(f"{LogColors.BOLD}{LogColors.HEADER}{msg}{LogColors.END}")

def get_input_action(env, oculus, cfg):
    """
    Get action from oculus controller
    """
    # prepare obs for oculus
    pose = env.unwrapped._robot.get_ee_pose()
    gripper = env.unwrapped._robot.get_gripper_position()
    state = {
        "robot_state": {
            "cartesian_position": pose,
            "gripper_position": gripper,
        }
    }

    vel_act, info =  oculus.forward(state, include_info=True, method="delta_action")
    # update_3d_plot(ax, info["delta_action"][:3], info["target_pos_offset"][:3], info["robot_pos_offset"][:3])
    
    # convert vel to delta actions
    delta_act = env.unwrapped._robot._ik_solver.cartesian_velocity_to_delta(
        vel_act
    )

    # prepare act
    if cfg.robot.DoF == 3:
        act = np.concatenate((delta_act[:3], vel_act[-1:]))
    elif cfg.robot.DoF == 4:
        act = np.concatenate((delta_act[:3], delta_act[5:6], vel_act[-1:]))
    elif cfg.robot.DoF == 6:
        act = np.concatenate((delta_act, vel_act[-1:]))
        
    if oculus.vr_state["gripper"] > 0.5:
        act[-1] = 0.5
    else:
        act[-1] = 0
    
    return act

@hydra.main(
    config_path="../../configs/", config_name="collect_demos_real_thinkpad", version_base="1.1"
)
def run_experiment(cfg):
    FLAGS(sys.argv)

    # configs
    log_config(f"language instruction: {cfg.language_instruction}")
    log_config(f"number of episodes: {cfg.episodes}")
    log_config(f"control hz: {cfg.robot.control_hz}")
    log_config(f"dataset name: {cfg.exp_id}")
    # No-ops related variables
    no_ops_threshold = cfg.no_ops_threshold
    mode = cfg.mode
    no_ops_last_detected_time = cfg.no_ops_last_detected_time
    savedir = f"{cfg.base_dir}/date_{date.today().month}{date.today().day}/{cfg.exp_id}"
    log_config(f"save directory: {savedir}")
    log_config(f"no_ops_threshold: {no_ops_threshold} secs")
    log_config(f"data collection mode: {mode}")
    
    # initialize env
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )
    env = DataCollectionWrapper(
        env,
        language_instruction=cfg.language_instruction,
        fake_blocking=False,
        act_noise_std=cfg.act_noise_std,
        save_dir=savedir,
    )
    env.reset_demo()

    log_success("Hold on... Resetting tthe robot...")
    
    # initialize oculus controller
    oculus = VRController(pos_action_gain=10, rot_action_gain=5) # sensitivity 
    assert oculus.get_info()["controller_on"], "ERROR: oculus controller off"
    log_success("Oculus Connected")

    while True:
        env.unwrapped.reset()

        log_instruction("Press 'A' to Start")
        # time to reset the scene
        while True:
            info = oculus.get_info()
            if info["success"]:
                # reset w/ recording obs after resetting the scene
                env.reset_demo()
                log_success("Tips: Hold the trigger to move the robot")
                # log_success("Tips 1: Hold the trigger to move the robot")
                break

        log_instruction("Press 'B' to reset the robot")
        while True:
            # wait for controller input
            info = oculus.get_info() 
            while (not info["success"] and not info["failure"]) and not info[
                "movement_enabled"
            ]:
                info = oculus.get_info()

            # press 'A' or 'B' to reset
            if info["success"] or info["failure"]:
                break
            
            # check if 'trigger' button is pressed
            if info["movement_enabled"]:
                act = get_input_action(env, oculus, cfg)
                env.step(act)


if __name__ == "__main__":
    run_experiment()