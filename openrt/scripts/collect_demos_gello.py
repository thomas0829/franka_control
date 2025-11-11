"""
GELLO Data Collection for Franka Robot
Combines:
  - teleop_gello_direct.py's proven GELLO control logic (joint position control)
  - collect_demos_real_oculus.py's data collection structure (DataCollectionWrapper)

Usage:
    sudo -E python openrt/scripts/collect_demos_gello.py \
        robot=real_7_dof_gello \
        robot.imgs=true \
        exp_id=pick_cube \
        episodes=10
"""
import os
import sys
import time
from datetime import date

import hydra
import numpy as np
import torch
from tqdm import tqdm
import pygame

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from robot.controllers.gello_controller import GelloController
from robot.wrappers.data_wrapper import DataCollectionWrapper
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict

# Setup logger (matching oculus version)
import logging

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


def trim_lowdim(obs):
    """
    將 lowdim_ee 裁回 Oculus 使用的 6D pose + gripper。
    """
    if obs is None:
        return obs
    obs = obs.copy()
    lowdim = obs.get("lowdim_ee")
    if isinstance(lowdim, np.ndarray) and lowdim.shape[-1] > 7:
        obs["lowdim_ee"] = lowdim[..., :7].copy()
    return obs


# ============================================================================
# Keyboard Interface
# ============================================================================

class KeyboardInterface:
    """Keyboard interface matching oculus VRController API."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 300))
        pygame.display.set_caption("GELLO Data Collection")
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.current_state = "READY"
        self.trajectory_count = 0
        self.total_trajectories = 0
        self._update_display()
    
    def _update_display(self):
        self.screen.fill((20, 20, 30))  # Dark blue background
        
        # State indicator with color
        state_colors = {
            "READY": (100, 100, 255),      # Blue
            "WAITING": (255, 255, 100),    # Yellow
            "COLLECTING": (100, 255, 100), # Green
            "SAVED": (0, 255, 0),          # Bright green
            "DISCARDED": (255, 100, 100),  # Red
        }
        color = state_colors.get(self.current_state, (255, 255, 255))
        
        # Draw state
        state_text = self.font_large.render(f"State: {self.current_state}", True, color)
        self.screen.blit(state_text, (20, 20))
        
        # Draw progress
        if self.total_trajectories > 0:
            progress_text = self.font_small.render(
                f"Progress: {self.trajectory_count}/{self.total_trajectories}", 
                True, (200, 200, 200)
            )
            self.screen.blit(progress_text, (20, 70))
        
        # Draw key hints
        hints = [
            ("S", "Start/Continue collecting", (100, 255, 100)),
            ("C", "Save trajectory (SUCCESS)", (100, 200, 255)),
            ("Q", "Discard trajectory (FAILURE)", (255, 100, 100)),
        ]
        
        y_offset = 120
        for key, description, key_color in hints:
            # Draw key with background
            key_bg = pygame.Rect(20, y_offset, 40, 35)
            pygame.draw.rect(self.screen, (50, 50, 60), key_bg)
            pygame.draw.rect(self.screen, key_color, key_bg, 2)
            
            key_text = self.font_large.render(key, True, key_color)
            self.screen.blit(key_text, (30, y_offset + 5))
            
            desc_text = self.font_small.render(description, True, (200, 200, 200))
            self.screen.blit(desc_text, (70, y_offset + 8))
            
            y_offset += 50
        
        pygame.display.flip()
    
    def set_state(self, state, traj_count=None, total_traj=None):
        """Update the current state display."""
        self.current_state = state
        if traj_count is not None:
            self.trajectory_count = traj_count
        if total_traj is not None:
            self.total_trajectories = total_traj
        self._update_display()
    
    def get_info(self):
        """Poll keyboard events (matching oculus.get_info() API)."""
        info = {
            'success': False,
            'failure': False,
            'movement_enabled': True,
            'controller_on': True,
        }
        
        pygame.event.pump()  # Process event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("Window closed")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # S = start/continue
                    info['success'] = True
                elif event.key == pygame.K_c:  # C = continue (also success)
                    info['success'] = True
                elif event.key == pygame.K_q:  # Q = quit/discard
                    info['failure'] = True
        
        return info
    
    def close(self):
        pygame.quit()


def get_input_action(env, gello, cfg):
    """
    Get action from GELLO controller using teleop_gello_direct.py's proven method
    
    Returns:
        act: Action array [7 joints + 1 gripper] for env.step()
    """
    # Read GELLO state (using teleop's proven method)
    gello_state = gello.get_joint_state()
    target_joints = gello_state[:7]
    gello_gripper = gello_state[7]  # 0=open, 1=closed
    
    # Prepare action (matching oculus format)
    act = np.append(target_joints, gello_gripper)
    
    return act


@hydra.main(
    config_path="../../configs/", 
    config_name="collect_demos_real_thinkpad", 
    version_base="1.1"
)
def run_experiment(cfg):
    """Main data collection function - combining teleop + oculus data collection."""
    
    cfg.robot.max_path_length = cfg.max_episode_length
    # Comment out for testing without camera
    # assert cfg.robot.imgs, "ERROR: set robot.imgs=true to record image observations!"
    
    # configs (matching oculus version)
    log_config(f"language instruction: {cfg.language_instruction}")
    log_config(f"number of episodes: {cfg.episodes}")
    log_config(f"max episode length: {cfg.max_episode_length}")
    log_config(f"control hz: {cfg.robot.control_hz}")
    log_config(f"dataset name: {cfg.exp_id}")
    savedir = f"{cfg.base_dir}/date_{date.today().month}{date.today().day}/{cfg.exp_id}"
    
    # Check if directory already exists
    pickle_dir = f"{savedir}_pickle"
    if os.path.exists(pickle_dir) and len(os.listdir(pickle_dir)) > 0:
        log_failure(f"\n{'='*70}")
        log_failure(f"⚠️  WARNING: Dataset directory already exists!")
        log_failure(f"   Path: {pickle_dir}")
        log_failure(f"   Contains {len(os.listdir(pickle_dir))} files")
        log_failure(f"{'='*70}")
        
        user_input = input("\nOptions:\n  [o] Overwrite (delete existing data)\n  [r] Rename (enter new name)\n  [q] Quit\nYour choice: ").strip().lower()
        
        if user_input == 'o':
            import shutil
            log_important(f"Deleting existing directory: {pickle_dir}")
            shutil.rmtree(pickle_dir)
            log_success("Existing data deleted. Starting fresh collection...")
        elif user_input == 'r':
            new_name = input("Enter new dataset name: ").strip()
            if new_name:
                cfg.exp_id = new_name
                savedir = f"{cfg.base_dir}/date_{date.today().month}{date.today().day}/{cfg.exp_id}"
                pickle_dir = f"{savedir}_pickle"
                log_success(f"Using new name: {cfg.exp_id}")
            else:
                log_failure("Invalid name. Exiting...")
                return
        else:
            log_failure("Exiting without collecting data.")
            return
    
    log_config(f"save directory: {savedir}")
    
    # No-ops related variables (matching oculus version)
    no_ops_threshold = cfg.no_ops_threshold
    mode = cfg.mode
    no_ops_last_detected_time = cfg.no_ops_last_detected_time
    log_config(f"no_ops_threshold: {no_ops_threshold} secs")
    log_config(f"data collection mode: {mode}")
    
    # initialize env (matching oculus version)
    log_connect("Initializing env...")
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
    obs = trim_lowdim(env.reset())
    log_success("Resetting env")
    
    # camera setup (matching oculus version)
    camera_names = [k for k in env.get_images().keys()]
    log_success(f"Initialized {len(camera_names)} camera(s): {camera_names}")
    
    # ========================================================================
    # Initialize GELLO controller (using teleop_gello_direct.py's method)
    # ========================================================================
    log_connect("Initializing GELLO controller...")
    gello_port = cfg.get('gello_port', '/dev/ttyUSB0')
    gello_baudrate = cfg.get('gello_baudrate', 57600)
    
    gello = GelloController(port=gello_port, baudrate=gello_baudrate)
    log_success("GELLO controller initialized")
    
    
    # Initial rough calibration (will re-calibrate when pressing 'S')
    log_instruction("⚙️  Initial calibration: Align GELLO with robot pose...")
    
    # Read initial robot state
    robot_joints = env.unwrapped._robot.get_joint_positions()
    
    # Read GELLO raw data
    gello_raw_rad = gello.driver.get_joints()
    gello_raw_rad = np.array(gello_raw_rad[:7])
    
    # Calculate offset (considering joint_signs)
    new_offset = gello_raw_rad - (robot_joints / gello.joint_signs)
    gello.joint_offsets = new_offset
    
    # Verify calibration
    gello_calibrated = gello.get_joint_state()[:7]
    diff = np.abs(gello_calibrated - robot_joints).max()
    
    if diff > 0.2:
        log_failure(f"⚠️  Warning: Large initial calibration error: {diff:.3f} rad ({diff*57.3:.1f}°)")
        log_instruction("Will re-calibrate when you press 'S' to start collecting")
    else:
        log_success(f"✅ Initial calibration OK (error: {diff:.3f} rad)")
    
    log_instruction("Note: Will auto-recalibrate when you press 'S' for each episode")
    
    # ========================================================================
    # Test cartesian control setup
    # ========================================================================
    log_instruction("\n🔧 Testing cartesian control setup...")
    try:
        import torch
        from utils.transformations import quat_to_euler, angle_diff
        
        # Get current robot state
        test_joints = env.unwrapped._robot.get_joint_positions()
        test_ee_pos = env.unwrapped._robot.get_ee_pos()
        test_ee_angle = env.unwrapped._robot.get_ee_angle()
        
        log_config(f"Current robot joints: {np.round(test_joints, 3).tolist()}")
        log_config(f"Current EE position: {np.round(test_ee_pos, 3).tolist()}")
        log_config(f"Current EE angle (rad): {np.round(test_ee_angle, 3).tolist()}")
        
        # Test delta calculation with GELLO (without FK for now)
        log_instruction("\n🔧 Testing GELLO calibration...")
        gello_state = gello.get_joint_state()
        gello_joints = gello_state[:7]
        gello_gripper = gello_state[7]
        
        log_config(f"GELLO joints: {np.round(gello_joints, 3).tolist()}")
        log_config(f"GELLO gripper: {gello_gripper:.3f}")
        
        # Check calibration (joints should be close)
        joint_diff = np.abs(gello_joints - test_joints)
        max_joint_diff = np.max(joint_diff)
        
        log_config(f"Joint differences: {np.round(joint_diff, 3).tolist()}")
        log_config(f"Max joint difference: {max_joint_diff:.3f} rad ({max_joint_diff*57.3:.1f}°)")
        
        if max_joint_diff < 0.2:  # ~11 degrees
            log_success("✅ GELLO calibration looks good!")
        else:
            log_failure(f"⚠️  Large joint difference detected!")
            log_failure("   GELLO might not be properly calibrated")
            log_instruction("   Please align GELLO with robot before collecting data")
        
        log_success("✅ Setup test completed!\n")
        log_instruction("📝 Note: Using cartesian delta control (matching Oculus data format)")
        log_instruction("   GELLO commands joints → Robot executes → Record actual cartesian delta\n")
        log_instruction("   This avoids FK computation and records what actually happened\n")
        
    except Exception as e:
        import traceback
        log_failure(f"❌ Setup test FAILED: {e}")
        log_failure(traceback.format_exc())
        user_confirm = input("Continue anyway? (y/n): ").strip().lower()
        if user_confirm != 'y':
            log_failure("Exiting...")
            return
    
    # Initialize keyboard interface
    keyboard = KeyboardInterface()
    assert keyboard.get_info()["controller_on"], "ERROR: keyboard controller off"
    log_success("Keyboard Interface Connected")
    
    # ========================================================================
    # Collection loop - EXACTLY matching oculus version structure
    # ========================================================================
    n_traj = int(cfg.start_traj)
    env.traj_count = n_traj
    interrupted = False  # Flag to track if collection was interrupted
    
    # Initialize pygame display with total trajectories
    keyboard.set_state("WAITING", traj_count=n_traj, total_traj=cfg.episodes)
    
    try:
        while n_traj < cfg.episodes:
            
            # reset w/o recording obs and w/o randomizing ee pos (matching oculus)
            randomize_ee_on_reset = env.unwrapped._randomize_ee_on_reset
            env.unwrapped._set_randomize_ee_on_reset(0.0)
            env.unwrapped.reset()
            env.unwrapped._set_randomize_ee_on_reset(randomize_ee_on_reset)
            
            log_instruction("Press 'S' to Start Collecting")
            keyboard.set_state("WAITING", traj_count=n_traj)
            
            # time to reset the scene
            while True:
                time.sleep(0.01)  # Small delay to prevent CPU spinning
                keyboard.set_state("WAITING", traj_count=n_traj)  # Keep updating display
                info = keyboard.get_info()
                if info["success"]:
                    # reset w/ recording obs after resetting the scene
                    obs = trim_lowdim(env.reset())
                    
                    # Re-calibrate GELLO for this episode
                    print("\n" + "="*70)
                    log_instruction("⚙️  AUTO-CALIBRATING GELLO FOR EPISODE {}...".format(n_traj))
                    print("="*70)
                    
                    robot_joints = obs['lowdim_qpos'][:7]  # Current robot joint positions
                    gello_raw_rad = gello.driver.get_joints()
                    gello_raw_rad = np.array(gello_raw_rad[:7])
                    new_offset = gello_raw_rad - (robot_joints / gello.joint_signs)
                    gello.joint_offsets = new_offset
                    
                    # Verify calibration
                    gello_calibrated = gello.get_joint_state()[:7]
                    diff = np.abs(gello_calibrated - robot_joints).max()
                    
                    print("="*70)
                    if diff < 0.2:
                        log_success("✅ CALIBRATION SUCCESSFUL! Error: {:.4f} rad ({:.2f}°)".format(diff, diff*57.3))
                    else:
                        log_failure(f"⚠️  WARNING: LARGE CALIBRATION ERROR: {diff:.4f} rad ({diff*57.3:.2f}°)")
                    print("="*70 + "\n")
                    
                    log_instruction("🎬 START COLLECTING TRAJECTORY {}".format(n_traj))
                    keyboard.set_state("COLLECTING", traj_count=n_traj)
                    break
            
            log_instruction("Press 'C' to Save (SUCCESS), Press 'Q' to Discard (FAILURE)")
            
            # no-ops related variables (matching oculus)
            first_no_ops_detected = True
            no_ops_start_time = 0
            save = False  # Initialize save flag outside the loop
            
            pbar = tqdm(
                range(cfg.max_episode_length),
                desc=f"Collecting Trajectory {n_traj}/{cfg.episodes}",
                leave=True,
                position=0,
            )
            
            for j in pbar:
                
                # Keep updating display during collection
                keyboard.set_state("COLLECTING", traj_count=n_traj)
                
                # wait for controller input (matching oculus)
                info = keyboard.get_info()
                
                while (not info["success"] and not info["failure"]) and not info["movement_enabled"]:
                    info = keyboard.get_info()
                
                # press '→' to indicate success (matching oculus)
                if info["success"]:
                    save = True
                    break  # Exit loop to save
                # press '←' to indicate failure (matching oculus)
                elif info["failure"]:
                    save = False
                    break  # Exit loop to discard
                
                # check if movement enabled (matching oculus)
                if info["movement_enabled"]:
                    
                    # ============================================================
                    # VERSION 1: Joint Control (Original - COMMENTED OUT)
                    # ============================================================
                    # # Get action using teleop_gello_direct.py's proven method
                    # act = get_input_action(env, gello, cfg)
                    # 
                    # # check if no-ops (matching oculus)
                    # if cfg.mode == "standard":
                    #     act_norm = np.linalg.norm(act)
                    #     
                    #     # first no-ops detected
                    #     if act_norm < no_ops_threshold and first_no_ops_detected:
                    #         first_no_ops_detected = False
                    #         no_ops_start_time = time.time()
                    #     # no-ops detected
                    #     elif act_norm < no_ops_threshold and not first_no_ops_detected:
                    #         if time.time() - no_ops_start_time >= no_ops_last_detected_time:
                    #             pbar.write(f"⚠️  No operation for over {round(time.time() - no_ops_start_time, 2)} secs")
                    #             break
                    #     # no-ops not detected (reset)
                    #     else:
                    #         first_no_ops_detected = True
                    #         no_ops_start_time = 0
                    # 
                    # # Use direct joint control (EXACTLY matching teleop_gello_direct.py)
                    # target_joints = act[:7]
                    # gello_gripper = act[7]
                    # 
                    # # Send commands directly to robot (bypassing env cartesian control)
                    # env.unwrapped._robot.update_joints(
                    #     target_joints.tolist(), velocity=False, blocking=False
                    # )
                    # env.unwrapped._robot.update_gripper(
                    #     gello_gripper, velocity=False, blocking=False
                    # )
                    # 
                    # # Sleep to maintain control frequency
                    # time.sleep(1.0 / cfg.robot.control_hz)
                    # 
                    # # Get observation after robot moves
                    # next_obs = env.unwrapped.get_observation()
                    # 
                    # # Manually record to DataCollectionWrapper buffer
                    # # (Matching DataCollectionWrapper.step() behavior)
                    # # Each buffer item contains: obs_t + language_instruction + action_t
                    # obs["language_instruction"] = cfg.language_instruction
                    # obs["action"] = act
                    # env.buffer.append(obs.copy())
                    # 
                    # # Update obs for next iteration
                    # obs = next_obs
                    # ============================================================
                    # VERSION 3: Joint Control + Measure Cartesian Delta
                    # (This is the method that worked for test_4!)
                    # ============================================================
                    from utils.transformations import angle_diff
                    
                    # Get GELLO target joint positions and gripper
                    gello_state = gello.get_joint_state()
                    target_joints = gello_state[:7]
                    gello_gripper = gello_state[7]
                    
                    # Get current EE pose (position + euler angles)
                    current_pos = obs["lowdim_ee"][:3]
                    current_angle = obs["lowdim_ee"][3:6]
                    
                    # Step 1: Move robot to GELLO's joint positions using joint control
                    # Use blocking=False for responsiveness
                    env.unwrapped._robot.update_joints(
                        target_joints.tolist(), velocity=False, blocking=False
                    )
                    env.unwrapped._robot.update_gripper(
                        gello_gripper, velocity=False, blocking=False
                    )
                    
                    # Sleep to maintain control frequency
                    time.sleep(1.0 / cfg.robot.control_hz)
                    
                    # Step 2: Get the new observation and calculate actual EE delta
                    next_obs = trim_lowdim(env.unwrapped.get_observation())
                    next_pos = next_obs["lowdim_ee"][:3]
                    next_angle = next_obs["lowdim_ee"][3:6]
                    
                    # Calculate delta (what actually happened in cartesian space)
                    delta_pos = next_pos - current_pos
                    delta_angle = angle_diff(next_angle, current_angle)
                    
                    # Prepare action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    act = np.concatenate((delta_pos, delta_angle, [gello_gripper]))
                    
                    # Step 3: Record the action to buffer (manually)
                    obs_entry = obs.copy()
                    obs_entry["language_instruction"] = cfg.language_instruction
                    obs_entry["action"] = act
                    env.buffer.append(obs_entry)
                    
                    # Update obs for next iteration
                    obs = next_obs
                    
                    # Safety check: detect abnormally large movements
                    delta_pos_norm = np.linalg.norm(delta_pos)
                    delta_angle_norm = np.linalg.norm(delta_angle)
                    
                    if delta_pos_norm > 0.1 or delta_angle_norm > 0.5:
                        pbar.write(f"⚠️  WARNING: Large movement detected!")
                        pbar.write(f"   Position delta: {delta_pos_norm:.4f}m, Angle delta: {delta_angle_norm:.4f}rad")
                        pbar.write(f"   Consider recalibrating GELLO.")
                    
                    # Check for no-ops (matching oculus)
                    if cfg.mode == "standard":
                        act_norm = np.linalg.norm(act[:6])  # Position + angle only
                        
                        if act_norm < no_ops_threshold and first_no_ops_detected:
                            first_no_ops_detected = False
                            no_ops_start_time = time.time()
                        elif act_norm < no_ops_threshold and not first_no_ops_detected:
                            if time.time() - no_ops_start_time >= no_ops_last_detected_time:
                                pbar.write(f"⚠️  No operation for over {round(time.time() - no_ops_start_time, 2)} secs")
                                break
                        else:
                            first_no_ops_detected = True
                            no_ops_start_time = 0
                    
                    # Update progress bar with current joint positions (instead of print)
                    if j % 10 == 0:  # Update every 10 steps to avoid too frequent updates
                        qpos = env.unwrapped._robot.get_joint_positions()
                        pbar.set_postfix({'joints': f'{qpos[:3].round(2).tolist()}...'}, refresh=False)
            
            # save trajectory if success (matching oculus)
            if save:
                keyboard.set_state("SAVED", traj_count=n_traj)
                env.save_buffer()
                n_traj += 1
                log_success("SUCCESS")
                time.sleep(0.5)  # Brief pause to show saved state
            else:
                keyboard.set_state("DISCARDED", traj_count=n_traj)
                # Reset buffer without saving (discard current trajectory)
                env.reset_buffer()
                log_failure("FAILURE")
                time.sleep(0.5)  # Brief pause to show discarded state
    
    except KeyboardInterrupt:
        log_failure("\n\nCollection interrupted by user")
        interrupted = True
        # Reset buffer to discard incomplete trajectory
        env.reset_buffer()
    
    finally:
        keyboard.close()
        
        # Only save/process if not interrupted
        if not interrupted:
            env.reset()
            log_success(f"Finished Collecting {n_traj} Trajectories")
        else:
            log_failure(f"Collection interrupted. Saved {n_traj} complete trajectories.")
        
        # ====================================================================
        # Auto-convert to HDF5 format (matching oculus workflow)
        # ====================================================================
        if n_traj > 0 and cfg.get('auto_convert_hdf5', True) and not interrupted:
            log_important("\n" + "="*70)
            log_important("Auto-converting to HDF5 format...")
            log_important("="*70)
            
            try:
                import glob
                import h5py
                import pickle
                import datetime
                
                # Prepare paths
                data_base_dir = f"{cfg.base_dir}/date_{date.today().month}{date.today().day}"
                pickle_dir = os.path.join(data_base_dir, f"{cfg.exp_id}_pickle")
                hdf5_output_dir = os.path.join(data_base_dir, f"{cfg.exp_id}_hdf5")
                os.makedirs(hdf5_output_dir, exist_ok=True)
                
                log_config(f"Input dir: {pickle_dir}")
                log_config(f"Output dir: {hdf5_output_dir}")
                
                # Create HDF5 file
                hdf5_file = h5py.File(os.path.join(hdf5_output_dir, "demos.hdf5"), "w")
                grp = hdf5_file.create_group("data")
                grp_mask = hdf5_file.create_group("mask")
                
                # Find all episode files (pickle format)
                file_names = sorted(glob.glob(os.path.join(pickle_dir, "*.pkl")))
                log_config(f"Found {len(file_names)} episodes to convert")
                
                demo_keys_train = []
                all_actions = []
                
                for i, file_name in enumerate(file_names):
                    log_config(f"Converting episode {i+1}/{len(file_names)}: {file_name}")
                    
                    # Load episode data (pickle format - already dict with arrays)
                    with open(file_name, 'rb') as f:
                        dic = pickle.load(f)
                    
                    # Extract actions
                    actions = dic["action"]
                    log_config(f"  Episode has {len(actions)} timesteps")
                    
                    # Create demo group
                    demo_key = f"demo_{i}"
                    demo_keys_train.append(demo_key)
                    ep_data_grp = grp.create_group(demo_key)
                    
                    # Add actions
                    ep_data_grp.create_dataset("actions", data=actions)
                    all_actions.append(actions)
                    
                    # Add dones
                    dones = np.zeros(len(actions)).astype(bool)
                    dones[-1] = True
                    ep_data_grp.create_dataset("dones", data=dones)
                    
                    # Create obs group
                    ep_obs_grp = ep_data_grp.create_group("obs")
                    
                    # Add all observations (excluding action)
                    for obs_key in dic.keys():
                        if obs_key == "action":
                            continue
                        obs = dic[obs_key]
                        if obs_key == "language_instruction":
                            obs = np.array(obs, dtype='S80')
                        ep_obs_grp.create_dataset(obs_key, data=obs)
                    
                    ep_data_grp.attrs["num_samples"] = len(actions)
                
                # Create mask dataset
                grp_mask.create_dataset("train", data=np.array(demo_keys_train, dtype="S"))
                
                # Add metadata
                grp.attrs["episodes"] = len(file_names)
                grp.attrs["env_args"] = '{"env_type": "franka", "type": "real"}'
                grp.attrs["type"] = "real"
                now = datetime.datetime.now()
                grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
                grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
                
                hdf5_file.close()
                
                log_success(f"✅ HDF5 conversion complete!")
                log_success(f"   Saved at: {hdf5_output_dir}/demos.hdf5")
                log_success(f"   Total episodes: {len(file_names)}")
                
                # ====================================================================
                # Create stats file with normalization bounds
                # ====================================================================
                log_important("\n" + "="*70)
                log_important("Computing action statistics...")
                log_important("="*70)
                
                if len(all_actions) > 0:
                    concat_actions = np.concatenate(all_actions, axis=0)
                    action_stats = {
                        "min": concat_actions.min(axis=0),
                        "max": concat_actions.max(axis=0),
                        "mean": concat_actions.mean(axis=0),
                        "std": concat_actions.std(axis=0) + 1e-8,
                    }
                else:
                    action_stats = {
                        "min": np.zeros(7),
                        "max": np.zeros(7),
                        "mean": np.zeros(7),
                        "std": np.ones(7),
                    }
                
                stats = {"action": action_stats, "normalized": False}
                
                stats_path = os.path.join(hdf5_output_dir, "stats")
                with open(stats_path, 'wb') as f:
                    pickle.dump(stats, f)
                
                log_success(f"✅ Stats file created!")
                log_success(f"   Saved at: {stats_path}")
                log_config(f"   Action min: {np.round(action_stats['min'], 4)}")
                log_config(f"   Action max: {np.round(action_stats['max'], 4)}")
                
            except Exception as e:
                import traceback
                log_failure(f"⚠️  HDF5 conversion failed: {e}")
                log_failure(traceback.format_exc())
                log_failure(f"   You can manually convert later using:")
                log_failure(f"   python openrt/scripts/convert_np_to_hdf5.py")


if __name__ == "__main__":
    run_experiment()
