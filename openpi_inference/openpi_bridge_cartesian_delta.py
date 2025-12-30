#!/usr/bin/env python3
"""
OpenPI-Polymetis Bridge
Connects OpenPI server (8000) to Franka robot via Polymetis (NUC)
Connects to existing robot services without re-launching them
"""
import time
import sys
import os
import numpy as np
import cv2
import pyrealsense2 as rs

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import OpenPI client
try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError:
    print("Error: openpi_client not found. Please install it:")
    print("  cd /home/duanj1/thomas/openpi/packages/openpi-client && pip install -e .")
    sys.exit(1)

# ============ Configuration ============
NUC_IP = "192.168.1.6"              # Polymetis zerorpc server IP (port 4242)
OPENPI_HOST = "127.0.0.1"           # OpenPI server IP
OPENPI_PORT = 5555                  # OpenPI server port
PROMPT = "Pick up the cube."

# RealSense camera serial numbers
EXT_SN   = "215222073684"           # External camera
WRIST_SN = "332322073412"           # Wrist camera
WIDTH, HEIGHT, FPS = 848, 480, 30   # Lower resolution for better performance (was 1280x720)
# 848x480 is native RealSense resolution, no scaling needed

# Control parameters
CTRL_HZ = 10.0                      # Control frequency (Hz) - match DROID dataset
MAX_TRANS = 0.01                    # Max translation (m)
MAX_ROT = 0.05                      # Max rotation (rad)
MAX_DQ = 0.15                       # Max joint delta (rad) - ~8.6 degrees
ACTION_SCALE = 0.3                  # Scale OpenPI actions (more conservative)
WAIT_AFTER_ACTION = 0.05            # Wait time after each action (sec) for motion to stabilize

# Gripper control parameters
GRIPPER_THRESHOLD = 0.5             # Threshold to distinguish open (>0.5) vs closed (<0.5)
GRIPPER_CHANGE_THRESHOLD = 0.3      # Minimum change to trigger gripper action (hysteresis)
GRIPPER_UPDATE_INTERVAL = 5         # Only check gripper every N steps (reduce jitter)

# DROID uses 10Hz control, slower is more stable
# =======================================


def open_color_pipeline(sn, width=848, height=480, fps=30):
    """Open RealSense color stream with optimized settings"""
    p = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    # Start pipeline
    profile = p.start(cfg)
    
    # Get the color sensor and optimize settings for performance
    device = profile.get_device()
    color_sensor = device.first_color_sensor()
    
    return p


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
            -0.02441366, -0.5955547, -0.0549833, -2.6330618, 
            -0.05316936, 2.06661602, 0.0
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
    print("\n[3/4] Initializing RealSense cameras...")
    try:
        ext_p = open_color_pipeline(EXT_SN, WIDTH, HEIGHT, FPS)
        print(f"✓ External camera (SN: {EXT_SN})")
        # Warm up camera (skip first few frames for auto-exposure to stabilize)
        for _ in range(10):
            ext_p.wait_for_frames()
    except Exception as e:
        print(f"✗ External camera connection failed: {e}")
        return
    
    try:
        wri_p = open_color_pipeline(WRIST_SN, WIDTH, HEIGHT, FPS)
        print(f"✓ Wrist camera (SN: {WRIST_SN})")
        # Warm up camera
        for _ in range(10):
            wri_p.wait_for_frames()
    except Exception as e:
        print(f"✗ Wrist camera connection failed: {e}")
        ext_p.stop()
        return
    
    # Preview cameras and wait for confirmation
    print("\n[4/5] Camera Preview")
    print("=" * 60)
    print("Displaying camera feeds...")
    print("Check the camera window to verify:")
    print("  - External camera shows the scene correctly")
    print("  - Wrist camera shows the robot gripper")
    print("  - Robot is at home position")
    print("  - Scene is set up correctly")
    print("=" * 60)
    print("\n👁️  Press ENTER when ready to start inference, or Ctrl+C to quit")
    
    try:
        # Preview loop - show cameras until user presses Enter
        preview_counter = 0
        while True:
            # Get images
            ext_f = ext_p.wait_for_frames().get_color_frame()
            wri_f = wri_p.wait_for_frames().get_color_frame()
            if not ext_f or not wri_f:
                continue
            
            ext_img = np.asanyarray(ext_f.get_data())
            wri_img = np.asanyarray(wri_f.get_data())
            
            # Only update display every 3 frames for smooth preview
            if preview_counter % 3 == 0:
                # Use smaller display size
                ext_display = cv2.resize(ext_img, (424, 240))
                wri_display = cv2.resize(wri_img, (424, 240))
                combined = np.hstack([ext_display, wri_display])
                
                # Add text labels (smaller for smaller window)
                cv2.putText(combined, "External", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(combined, "Wrist", (440, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(combined, "Press ENTER to start", (200, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
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
                ext_p.stop()
                wri_p.stop()
                return
                
    except KeyboardInterrupt:
        print("\n✗ Cancelled by user")
        cv2.destroyAllWindows()
        ext_p.stop()
        wri_p.stop()
        return
    
    dt = 1.0 / CTRL_HZ
    print(f"\n[5/5] Starting control loop")
    print(f"Frequency: {CTRL_HZ} Hz")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    step = 0
    last_print_time = time.time()
    
    # Gripper state tracking (to prevent jitter)
    last_gripper_state = None  # None, "OPEN", or "CLOSED"
    last_gripper_value = None  # Last OpenPI gripper value
    last_gripper_cmd = 0.0     # Last command sent to robot (Robotiq format)
    
    try:
        while True:
            t0 = time.time()
            
            # Get images (fast, no processing)
            ext_f = ext_p.wait_for_frames().get_color_frame()
            wri_f = wri_p.wait_for_frames().get_color_frame()
            if not ext_f or not wri_f:
                continue
            
            ext_img = np.asanyarray(ext_f.get_data())
            wri_img = np.asanyarray(wri_f.get_data())
            
            # No display during inference (window closed after Enter)
            # Images are still captured for OpenPI inference
            
            # Get current robot state
            joint_pos = robot.get_joint_positions()
            gripper_pos = robot.get_gripper_position()
            
            # Prepare observation
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
            
            # Execute action
            # OpenPI returns: {"actions": (10, 8)} where each action is [dq1...dq7, gripper]
            try:
                if "actions" not in out:
                    print(f"✗ No 'actions' key in output: {out.keys()}")
                    break
                
                actions = out["actions"]  # shape: (horizon, 8)
                if len(actions.shape) != 2 or actions.shape[1] != 8:
                    print(f"✗ Unexpected action shape: {actions.shape}, expected (N, 8)")
                    break
                
                # Use first action (index 0)
                action = actions[0]  # shape: (8,)
                dq_raw = action[:7]  # joint deltas (raw from model)
                gripper = action[7]  # gripper command
                
                # Scale the actions (OpenPI may output larger values)
                dq_scaled = dq_raw * ACTION_SCALE
                
                # Clip joint deltas for safety
                dq = np.clip(dq_scaled, -MAX_DQ, MAX_DQ)
                
                # Debug: show raw vs scaled vs clipped values
                if step % 10 == 0:
                    print(f"  Raw: max={np.abs(dq_raw).max():.3f}, "
                          f"Scaled: max={np.abs(dq_scaled).max():.3f}, "
                          f"Clipped: max={np.abs(dq).max():.3f}")
                
                # Get current joint positions and apply delta
                q_current = robot.get_joint_positions()
                q_target = q_current + dq
                
                # Send joint command (blocking for smoother execution)
                # At 10Hz, blocking is fine and gives more stable motion
                robot.update_joints(
                    command=q_target.tolist(),
                    velocity=False,
                    blocking=True
                )
                
                # Small wait for motion to stabilize
                time.sleep(WAIT_AFTER_ACTION)
                
                # Check if motion was executed
                q_after = robot.get_joint_positions()
                actual_dq = q_after - q_current
                execution_ratio = np.linalg.norm(actual_dq) / (np.linalg.norm(dq) + 1e-6)
                
                if step % 10 == 0 and execution_ratio < 0.5:
                    print(f"  ⚠ Action underexecuted: commanded {np.linalg.norm(dq):.4f}, "
                          f"actual {np.linalg.norm(actual_dq):.4f} (ratio: {execution_ratio:.2f})")
                
                # ===== Intelligent Gripper Control =====
                # Only update gripper when there's a significant state change
                # This prevents jitter from small fluctuations in the model output
                
                gripper_openpi = np.clip(gripper, 0.0, 1.0)
                
                # Determine desired gripper state based on threshold
                desired_state = "OPEN" if gripper_openpi > GRIPPER_THRESHOLD else "CLOSED"
                
                # Only update gripper if:
                # 1. State has changed (OPEN <-> CLOSED transition)
                # 2. OR it's a gripper update interval
                # 3. OR this is the first step
                should_update_gripper = False
                
                if last_gripper_state is None:
                    # First step - initialize gripper
                    should_update_gripper = True
                    print(f"  🔧 Initializing gripper: {desired_state}")
                elif desired_state != last_gripper_state:
                    # State transition detected
                    # Only act if the change is significant (hysteresis)
                    if last_gripper_value is not None:
                        change = abs(gripper_openpi - last_gripper_value)
                        if change > GRIPPER_CHANGE_THRESHOLD:
                            should_update_gripper = True
                            print(f"  🔧 Gripper state change: {last_gripper_state} -> {desired_state} (Δ={change:.3f})")
                elif step % GRIPPER_UPDATE_INTERVAL == 0 and step > 0:
                    # Periodic update to ensure gripper maintains position
                    # Only if value has changed significantly
                    if last_gripper_value is not None:
                        change = abs(gripper_openpi - last_gripper_value)
                        if change > 0.1:  # Smaller threshold for maintenance updates
                            should_update_gripper = True
                
                if should_update_gripper:
                    # Convert to Robotiq format (inverted)
                    gripper_cmd = 1.0 - gripper_openpi
                    robot.update_gripper(
                        command=gripper_cmd,
                        velocity=False,
                        blocking=False
                    )
                    last_gripper_state = desired_state
                    last_gripper_value = gripper_openpi
                    last_gripper_cmd = gripper_cmd
                else:
                    # Skip gripper update (use last command)
                    gripper_cmd = last_gripper_cmd
                
                # ===== End Gripper Control =====
                
                # Print progress
                if step % 10 == 0:
                    # Show current gripper state
                    grip_display = f"{last_gripper_state}({gripper_openpi:.2f})"
                    
                    # Get end-effector position for spatial tracking
                    try:
                        ee_pos = robot.get_ee_pose()[:3, 3]  # Extract xyz position
                        print(f"Step {step:4d}: δq=[{dq[0]:+.3f},{dq[1]:+.3f},{dq[2]:+.3f},{dq[3]:+.3f},"
                              f"{dq[4]:+.3f},{dq[5]:+.3f},{dq[6]:+.3f}] "
                              f"EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
                              f"grip={grip_display}")
                    except:
                        # Fallback if get_ee_pose doesn't work
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
            # Use the same home position as robot_env.py
            home_joints = np.array([
                -0.02441366, -0.5955547, -0.0549833, -2.6330618, 
                -0.05316936, 2.06661602, 0.0
            ])
            
            # Move to home position
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
            print("  Opening gripper...")
            robot.update_gripper(command=0.0, velocity=False, blocking=True)
            print("  ✓ Gripper opened")
            
            print("✓ Robot reset complete")
        except Exception as e:
            print(f"  ⚠ Failed to reset robot: {e}")
        
        # Close cameras and windows
        cv2.destroyAllWindows()
        ext_p.stop()
        wri_p.stop()
        print("✓ Cameras and display closed")
        print(f"✓ Total steps: {step}")
        print("Program ended")


if __name__ == "__main__":
    main()
