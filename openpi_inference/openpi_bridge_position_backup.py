#!/usr/bin/env python3
"""
OpenPI Joint Position - Polymetis Bridge
Connects OpenPI Joint Position server (8000) to Franka robot via Polymetis (NUC)

This version is for models that output ABSOLUTE joint positions (not velocities).

Supported Models:
  - pi0_droid_jointpos: Original PI0 position model (action_horizon=10)
  - pi05_droid_jointpos: PI05 position model (action_horizon=15)
  - pi0_fast_droid_jointpos: Fast PI0 position variant

Usage:
  1. Change MODEL_TYPE in Configuration section below
  2. Start model server on port 8000:
     - PI0:  python scripts/serve_policy.py --policy.config=pi0_droid --policy.dir=./pi0_droid_jointpos
     - PI05: python scripts/serve_policy.py --policy.config=pi05_droid_jointpos --policy.dir=./pi05_droid_jointpos
  3. Run this bridge: python openpi_bridge_position.py

Model output format: [action_horizon, 8] where each action is [q1...q7, gripper]
  - q1-q7: Absolute joint positions in radians
  - gripper: 0.0 (closed) to 1.0 (open)
"""
import time
import sys
import os
import signal
import atexit
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

# Excel logging imports (only used if ENABLE_EXCEL_LOGGING is True)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Global flag for emergency stop
_emergency_stop = False

def signal_handler(signum, frame):
    """Handle Ctrl+C by setting emergency stop flag"""
    global _emergency_stop
    _emergency_stop = True
    print("\n\n🛑 Ctrl+C detected - Emergency stop triggered!", flush=True)
    raise KeyboardInterrupt  # Raise to interrupt input() calls

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import OpenPI client
try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError:
    print("Error: openpi_client not found. Please install it:")
    print("  cd /home/duanj1/thomas/openpi/packages/openpi-client && pip install -e .")
    sys.exit(1)

# Import ZED SDK
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    print("Warning: ZED SDK not found. ZED camera will not be available.")
    ZED_AVAILABLE = False

# ============ Configuration ============
# Model Selection: "pi0_droid_jointpos" or "pi05_droid_jointpos"
MODEL_TYPE = "pi05_droid_jointpos"  # Using PI0.5 (works for grasping)

NUC_IP = "192.168.1.6"              # Polymetis zerorpc server IP (port 4242)
OPENPI_HOST = "127.0.0.1"           # OpenPI server IP
OPENPI_PORT = 8000                  # OpenPI Joint Position server port (default 8000)
PROMPT = "Put the cube on the plate."

# ============ Loop Recording Configuration ============
# Set LOOP to the number of episodes you want to record
# LOOP = 0: Stop record (inference only, no video)
# LOOP = 1: Record 1 episode (single recording)
# LOOP = 3: Record 3 episodes (will loop 3 times)
# Each episode will be saved in a separate timestamped folder under a task-named directory
LOOP = 3  # Number of episodes to record

# ============ Excel Logging Configuration ============
# Set to True to enable Excel logging of episode metadata
# When enabled, you will be prompted after each episode for:
# - Model name (e.g., "pi0", "pi05") - asked once at start
# - Success status (y/n) - asked after each episode
# Data logged: Task, Episode, Model, Success, Video Path, Steps
# Excel file will be saved in the task directory as "episode_log.xlsx"
ENABLE_EXCEL_LOGGING = True  # True: Enable logging, False: Disable

# ZED Camera Configuration (Official OpenPI DROID Setup)
ZED_EXTERNAL_ID = 0                 # ZED 2 (external/shoulder view) - SN: 26706125
ZED_WRIST_ID = 1                    # ZED Mini (wrist view) - SN: 14943057
WIDTH, HEIGHT, FPS = 1280, 720, 15  # ZED HD720 mode @ 15fps (official DROID config)

# Control parameters
CTRL_HZ = 10.0                      # Control frequency (Hz) - match DROID

# Model-specific settings (PI0 vs PI05 have different action horizons)
# PI0: action_horizon=10 (outputs 10 actions per query, shape [10, 8])
#      WARNING: PI0 outputs DELTA positions (incremental changes), NOT absolute!
# PI05: action_horizon=16 (outputs 16 actions per query, shape [16, 8])
#       PI05 outputs ABSOLUTE positions
if "pi0" in MODEL_TYPE.lower() and "pi05" not in MODEL_TYPE.lower():
    OPEN_LOOP_HORIZON = 10          # PI0: use all 10 actions from model
    MAX_DQ = 0.10                   # PI0: limit per-step delta
    OUTPUT_IS_DELTA = True          # PI0 outputs position deltas
    USE_EMA_SMOOTHING = False       # Don't smooth deltas - use them directly
    DELTA_SCALE = 0.3               # Scale down to 30% (0.3 is safe, 1.0 crashes into table)
else:  # pi05
    OPEN_LOOP_HORIZON = 8           # PI05: use first 8 of 16 actions (works well)
    MAX_DQ = 0.25                   # PI05: can handle larger movements
    OUTPUT_IS_DELTA = False         # PI05 outputs absolute positions
    USE_EMA_SMOOTHING = True        # Smooth absolute positions to reduce jumps
    DELTA_SCALE = 1.0               # No scaling for PI05 (uses absolute positions)

EMA_ALPHA = 0.3                     # EMA smoothing factor (0.3 = 30% new, 70% old) - only used if USE_EMA_SMOOTHING=True
# NOTE: Joint Position model outputs ABSOLUTE positions, not deltas
# We calculate delta for safety checking only

# Maximum steps before stopping each episode
MAX_STEPS = 1000

print(f"Model: {MODEL_TYPE}")
print(f"Control frequency: {CTRL_HZ} Hz")
print(f"Open loop horizon: {OPEN_LOOP_HORIZON} actions")
print(f"Max joint delta: {MAX_DQ} rad")
print(f"Max steps per episode: {MAX_STEPS}")
print(f"Output type: {'DELTA (incremental)' if OUTPUT_IS_DELTA else 'ABSOLUTE (positions)'}")
if OUTPUT_IS_DELTA:
    print(f"Delta scale: {DELTA_SCALE:.1f} (SAFETY: scaling down delta for testing)")
print(f"EMA smoothing: {'enabled (alpha=' + str(EMA_ALPHA) + ')' if USE_EMA_SMOOTHING else 'disabled'}")
# =======================================


# Global camera references for cleanup
_zed_ext = None
_zed_wri = None

def cleanup_cameras():
    """Cleanup function to ensure cameras are properly closed"""
    global _zed_ext, _zed_wri
    
    print("\nCleaning up cameras...")
    try:
        if _zed_ext is not None:
            _zed_ext.close()
            print("  ✓ External ZED camera closed")
            _zed_ext = None
    except Exception as e:
        print(f"  ⚠ Error closing external camera: {e}")
    
    try:
        if _zed_wri is not None:
            _zed_wri.close()
            print("  ✓ Wrist ZED camera closed")
            _zed_wri = None
    except Exception as e:
        print(f"  ⚠ Error closing wrist camera: {e}")
    
    # Add a small delay to ensure cleanup completes
    time.sleep(0.5)

def signal_handler(sig, frame):
    """Handle termination signals"""
    print(f"\n\nReceived signal {sig}")
    cleanup_cameras()
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_cameras)
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command


def open_zed_camera(camera_id=None, serial_number=None, width=1280, height=720, fps=15):
    """Open ZED camera with optimized settings
    
    Args:
        camera_id: Camera ID (0 for first camera, 1 for second, etc.)
        serial_number: Camera serial number (alternative to camera_id)
        width, height: Resolution (1280x720 for HD720)
        fps: Frame rate (15 fps for DROID)
    """
    if not ZED_AVAILABLE:
        raise RuntimeError("ZED SDK is not available. Please install pyzed.")
    
    if camera_id is None and serial_number is None:
        raise ValueError("Must provide either camera_id or serial_number")
    
    zed = sl.Camera()
    
    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 1280x720
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # We only need RGB, no depth
    init_params.coordinate_units = sl.UNIT.METER
    
    # Set camera ID or serial number
    if camera_id is not None:
        init_params.set_from_camera_id(camera_id)
        identifier = f"ID {camera_id}"
    else:
        init_params.set_from_serial_number(serial_number)
        identifier = f"SN {serial_number}"
    
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera {identifier}: {err}")
    
    # Get camera information
    cam_info = zed.get_camera_information()
    camera_model = cam_info.camera_model
    actual_serial_number = cam_info.serial_number
    
    # Set camera controls for better image quality
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)  # Enable auto white balance
    
    # Set runtime parameters
    runtime_params = sl.RuntimeParameters()
    
    return zed, runtime_params, camera_model, actual_serial_number


def get_zed_image(zed, runtime_params, rotate_180=False):
    """Get RGB image from ZED camera
    
    Args:
        zed: ZED camera object
        runtime_params: Runtime parameters
        rotate_180: Whether to rotate image 180 degrees (for upside-down mounting)
    """
    image = sl.Mat()
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # Convert BGRA to BGR
        img_bgra = image.get_data()
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        # Rotate if needed (e.g., wrist camera mounted upside down)
        if rotate_180:
            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)
        return img_bgr
    else:
        return None


def bgr_to_rgb(img):
    """Convert BGR to RGB"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    print("=" * 60)
    print("OpenPI-Polymetis Bridge")
    print("=" * 60)
    print(f"NUC IP: {NUC_IP}:4242 (zerorpc)")
    print(f"OpenPI: {OPENPI_HOST}:{OPENPI_PORT}")
    print(f"Task: {PROMPT}")
    print("=" * 60)
    
    # Connect to OpenPI server
    print("\n[1/4] Connecting to OpenPI server...")
    try:
        openpi_client = WebsocketClientPolicy(host=OPENPI_HOST, port=OPENPI_PORT)
        server_metadata = openpi_client.get_server_metadata()
        print(f"✓ OpenPI connected")
        print(f"  Server metadata: {server_metadata}")
    except Exception as e:
        print(f"✗ OpenPI connection failed: {e}")
        print("Make sure OpenPI server is running on port 5555")
        return
    
    # Connect to robot (same as GELLO: launch=True will auto-start robot on NUC)
    print("\n[2/4] Connecting to robot on NUC...")
    try:
        from robot.real.server_interface import ServerInterface
        
        # Connect to zerorpc and auto-launch robot (same as GELLO)
        print(f"  Connecting to zerorpc at {NUC_IP}:4242...")
        print("  (This will auto-launch Polymetis robot_server on NUC)")
        robot = ServerInterface(ip_address=NUC_IP)  # launch=True by default!
        print("  ✓ Robot launched and connected")
        
        # Test connection by getting robot state
        print("  Testing robot state...")
        joint_pos = robot.get_joint_positions()
        print(f"  ✓ Got joint positions: {joint_pos[:3]}... (showing first 3)")
        
        # Reset to home position at startup
        print("\n  Resetting robot to home position...")
        home_joints = np.array([
            0.0, -0.50, 0.0, -2.40, 
            0.0, 1.90, 0.0
        ])
        current_joints = robot.get_joint_positions()
        print(f"    Current: {np.round(current_joints, 3).tolist()}")
        print(f"    Target:  {np.round(home_joints, 3).tolist()}")
        
        # Move to home position
        robot.update_joints(
            command=home_joints.tolist(),
            velocity=False,
            blocking=True
        )
        
        # Open gripper (0.0 = open for Robotiq)
        robot.update_gripper(command=0.0, velocity=False, blocking=True)
        print("  ✓ Robot reset to home position")
        
        print("✓ Robot interface ready")
        
    except Exception as e:
        print(f"✗ Robot connection failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Open cameras
    print(f"\n[3/4] Initializing ZED cameras...")
    print("Camera setup: 2x ZED cameras (official OpenPI configuration)")
    print(f"  - ZED 2 (external/shoulder view): Camera ID {ZED_EXTERNAL_ID}")
    print(f"  - ZED Mini (wrist view): Camera ID {ZED_WRIST_ID}")
    
    # Initialize variables
    global _zed_ext, _zed_wri
    zed_ext = None
    zed_ext_runtime = None
    zed_wri = None
    zed_wri_runtime = None
    
    # Initialize external ZED camera (ZED 2)
    try:
        zed_ext, zed_ext_runtime, ext_model, ext_sn = open_zed_camera(
            camera_id=ZED_EXTERNAL_ID,
            width=WIDTH,
            height=HEIGHT,
            fps=FPS
        )
        _zed_ext = zed_ext  # Store for cleanup
        print(f"✓ External ZED camera (ID: {ZED_EXTERNAL_ID})")
        print(f"  Model: {ext_model}, SN: {ext_sn}")
        # Warm up camera
        for _ in range(10):
            get_zed_image(zed_ext, zed_ext_runtime)
    except Exception as e:
        print(f"✗ External ZED camera initialization failed: {e}")
        cleanup_cameras()
        return
    
    # Initialize wrist ZED camera (ZED Mini)
    try:
        zed_wri, zed_wri_runtime, wri_model, wri_sn = open_zed_camera(
            camera_id=ZED_WRIST_ID,
            width=WIDTH,
            height=HEIGHT,
            fps=FPS
        )
        _zed_wri = zed_wri  # Store for cleanup
        print(f"✓ Wrist ZED camera (ID: {ZED_WRIST_ID})")
        print(f"  Model: {wri_model}, SN: {wri_sn}")
        # Warm up camera
        for _ in range(10):
            get_zed_image(zed_wri, zed_wri_runtime)  # No rotation
    except Exception as e:
        print(f"✗ Wrist ZED camera initialization failed: {e}")
        cleanup_cameras()
        return
    
    # Preview cameras and wait for confirmation
    print("\n[4/5] Camera Preview")
    print("=" * 60)
    print("Displaying camera feeds...")
    print("Check the camera window to verify:")
    print("  - External ZED 2 (left) shows the scene correctly")
    print("  - Wrist ZED Mini (right) shows the robot gripper")
    print("  - Robot is at home position")
    print("  - Scene is set up correctly")
    print("=" * 60)
    print("\n👁️  Press ENTER when ready to start inference, or Ctrl+C to quit")
    
    try:
        # Preview loop - show cameras until user presses Enter
        preview_counter = 0
        while True:
            # Collect active camera images
            cameras_to_show = []
            camera_labels = []
            
            # Get External ZED image
            ext_img = None
            if zed_ext:
                ext_img = get_zed_image(zed_ext, zed_ext_runtime)
                if ext_img is not None:
                    cameras_to_show.append(ext_img)
                    camera_labels.append("External (ZED 2)")
            
            # Get Wrist ZED image
            wri_img = None
            if zed_wri:
                wri_img = get_zed_image(zed_wri, zed_wri_runtime)  # No rotation
                if wri_img is not None:
                    cameras_to_show.append(wri_img)
                    camera_labels.append("Wrist (ZED Mini)")
            
            # Skip if no wrist image available
            if wri_img is None or ext_img is None:
                continue
            
            # Only update display every 3 frames for smooth preview
            if preview_counter % 3 == 0 and len(cameras_to_show) > 0:
                # Resize all cameras (doubled size: 848x480 instead of 424x240)
                displays = [cv2.resize(img, (848, 480)) for img in cameras_to_show]
                
                # Stack horizontally
                combined = np.hstack(displays)
                
                # Add text labels (larger font for bigger window)
                for i, label in enumerate(camera_labels):
                    x_pos = 20 + i * 860
                    cv2.putText(combined, label, (x_pos, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Add instruction text (centered based on number of cameras)
                text_x = 280 if len(cameras_to_show) == 2 else 600
                cv2.putText(combined, "Press ENTER to start", (text_x, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                cv2.imshow("Camera Preview - Press ENTER to start", combined)
            
            preview_counter += 1
            
            # Check for Enter key or window close (check every frame for responsiveness)
            key = cv2.waitKey(1)  # 1ms instead of 30ms for better responsiveness
            if key == 13:  # Enter key
                print("\n✓ Starting inference...")
                cv2.destroyAllWindows()  # Close the preview window
                break
            elif key == 27:  # ESC key
                print("\n✗ Cancelled by user")
                cv2.destroyAllWindows()
                cleanup_cameras()
                return
                
    except KeyboardInterrupt:
        print("\n✗ Cancelled by user")
        cv2.destroyAllWindows()
        cleanup_cameras()
        return
    
    dt = 1.0 / CTRL_HZ
    print(f"\n[5/5] Starting control loop")
    print(f"Frequency: {CTRL_HZ} Hz")
    print(f"Open loop horizon: {OPEN_LOOP_HORIZON} steps (query model every {OPEN_LOOP_HORIZON} steps)")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    step = 0
    last_print_time = time.time()
    
    # Open loop control: cache model predictions
    action_queue = []  # Queue of actions to execute
    
    # EMA smoothing: track smoothed target position
    q_target_smoothed = None  # Will be initialized on first step
    
    try:
        while True:
            t0 = time.time()
            
            # Query model when action queue is empty
            if len(action_queue) == 0:
                # Get images from ZED cameras
                ext_img = get_zed_image(zed_ext, zed_ext_runtime)
                wri_img = get_zed_image(zed_wri, zed_wri_runtime)  # No rotation
                
                if ext_img is None or wri_img is None:
                    continue
                
                # Get current robot state
                joint_pos = robot.get_joint_positions()
                gripper_pos = robot.get_gripper_position()
                
                # Prepare observation (convert BGR to RGB)
                obs = {
                    "observation/exterior_image_1_left": bgr_to_rgb(ext_img),
                    "observation/wrist_image_left": bgr_to_rgb(wri_img),
                    "observation/joint_position": joint_pos,
                    "observation/gripper_position": gripper_pos,
                    "prompt": PROMPT,
                }
                
                # Get action from OpenPI
                try:
                    out = openpi_client.infer(obs)
                except Exception as e:
                    if time.time() - last_print_time > 1.0:
                        print(f"⚠ OpenPI inference error: {e}")
                        last_print_time = time.time()
                    time.sleep(dt)
                    continue
                
                # Parse actions
                if "actions" not in out:
                    print(f"✗ No 'actions' key in output: {out.keys()}")
                    break
                
                actions = out["actions"]  # shape: (horizon, 8) - PI0: (10,8), PI05: (16,8)
                if len(actions.shape) != 2 or actions.shape[1] != 8:
                    print(f"✗ Unexpected action shape: {actions.shape}, expected (N, 8)")
                    break
                
                # Use first OPEN_LOOP_HORIZON actions
                horizon = min(OPEN_LOOP_HORIZON, len(actions))
                action_queue = list(actions[:horizon])
                
                if step % 10 == 0:
                    print(f"  📥 Queried model: got {len(actions)} actions, using first {horizon}")
            
            # Execute next action from queue
            try:
                action = action_queue.pop(0)  # Get and remove first action
                q_model_output = action[:7]  # Joint output from model (either delta or absolute)
                gripper = action[7]  # gripper command
                
                # Get current joint positions
                q_current = robot.get_joint_positions()
                
                # ===== Process Model Output (Delta vs Absolute) =====
                if OUTPUT_IS_DELTA:
                    # PI0: Model outputs position DELTA (incremental change)
                    dq_raw = q_model_output * DELTA_SCALE  # Scale down delta for safety
                    q_target_raw = q_current + dq_raw  # Calculate target position
                    
                    # No EMA smoothing for deltas - use them directly
                    dq = dq_raw
                    q_target_smoothed = q_target_raw
                else:
                    # PI05: Model outputs ABSOLUTE position
                    q_target_raw = q_model_output  # Output IS the target position
                    
                    if USE_EMA_SMOOTHING:
                        # Apply EMA smoothing to absolute positions
                        if q_target_smoothed is None:
                            q_target_smoothed = q_target_raw.copy()
                        else:
                            q_target_smoothed = EMA_ALPHA * q_target_raw + (1 - EMA_ALPHA) * q_target_smoothed
                    else:
                        q_target_smoothed = q_target_raw
                    
                    # Calculate delta from smoothed target
                    dq_raw = q_target_raw - q_current
                    dq = q_target_smoothed - q_current
                
                # Safety check: clip large movements
                dq_clipped = np.clip(dq, -MAX_DQ, MAX_DQ)
                
                # Use clipped delta
                q_target = q_current + dq_clipped
                
                # Debug: show position and delta
                if step % 10 == 0:
                    if OUTPUT_IS_DELTA:
                        print(f"  Current: {q_current[:3].round(3)}... Delta: {dq_raw[:3].round(3)}... Target: {q_target[:3].round(3)}...")
                    else:
                        print(f"  Current: {q_current[:3].round(3)}... Raw: {q_target_raw[:3].round(3)}... Smoothed: {q_target_smoothed[:3].round(3)}...")
                    print(f"  Delta magnitude: {np.abs(dq).max():.3f}, clipped: {np.abs(dq_clipped).max():.3f}")
                    print(f"  Actions remaining in queue: {len(action_queue)}")
                
                # Send joint command (non-blocking for faster response)
                robot.update_joints(
                    command=q_target.tolist(),
                    velocity=False,
                    blocking=False
                )
                
                # ===== Gripper Control (Continuous with threshold) =====
                # Use continuous gripper values directly (0=closed, 1=open)
                gripper_openpi = np.clip(gripper, 0.0, 1.0)
                
                # Apply threshold: values below 0.3 become 0 (closed)
                if gripper_openpi < 0.3:
                    gripper_action = 0.0  # Force closed
                else:
                    gripper_action = gripper_openpi  # Use continuous value
                
                # Try NOT inverting - use direct value
                gripper_cmd = gripper_action
                
                # Send to robot
                robot.update_gripper(command=gripper_cmd, velocity=False, blocking=False)
                
                # ===== End Gripper Control =====
                
                # Print progress
                if step % 10 == 0:
                    grip_display = f"{gripper_openpi:.2f}"  # Show continuous value
                    
                    try:
                        ee_pose = robot.get_ee_pose()
                        ee_pos = ee_pose[:3]
                        print(f"Step {step:4d}: δq=[{dq[0]:+.3f},{dq[1]:+.3f},{dq[2]:+.3f},{dq[3]:+.3f},"
                              f"{dq[4]:+.3f},{dq[5]:+.3f},{dq[6]:+.3f}] "
                              f"EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
                              f"grip={grip_display}")
                    except:
                        print(f"Step {step:4d}: δq=[{dq[0]:+.3f},{dq[1]:+.3f},{dq[2]:+.3f},{dq[3]:+.3f},"
                              f"{dq[4]:+.3f},{dq[5]:+.3f},{dq[6]:+.3f}] grip={grip_display}")
            
            except Exception as e:
                print(f"✗ Action execution error: {e}")
                import traceback
                traceback.print_exc()
                break
            
            step += 1
            elapsed = time.time() - t0
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
            
            # Monitor loop time
            if elapsed > dt * 1.5 and step % 10 == 0:
                print(f"⚠ Loop time too long: {elapsed*1000:.1f}ms (target: {dt*1000:.1f}ms)")
            
    except KeyboardInterrupt:
        print("\n\nReceived stop signal")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        
        # Reset robot to home position (same as GELLO)
        print("  Resetting robot to home position...")
        try:
            # Compromise position - balances model expectations with safety
            home_joints = np.array([
                0.0, -0.50, 0.0, -2.40, 
                0.0, 1.90, 0.0
            ])
            
            current_joints = robot.get_joint_positions()
            print(f"    Current: {np.round(current_joints, 3).tolist()}")
            print(f"    Target:  {np.round(home_joints, 3).tolist()}")
            
            # Smooth move to home (blocking)
            robot.update_joints(
                command=home_joints.tolist(),
                velocity=False,
                blocking=True
            )
            print("  ✓ Moved to home position")
            
            # Open gripper (0.0 = open for Robotiq)
            robot.update_gripper(command=0.0, velocity=False, blocking=True)
            print("  ✓ Gripper opened")
            
            print("✓ Robot reset complete")
        except Exception as e:
            print(f"  ⚠ Failed to reset robot: {e}")
        
        # Close cameras and windows
        cv2.destroyAllWindows()
        cleanup_cameras()
        print("✓ Cameras and display closed")
        print(f"✓ Total steps: {step}")
        print("Program ended")


if __name__ == "__main__":
    main()
