"""
GELLO Controller for Franka Robot
Replaces VRController (Oculus) with GELLO hardware input
"""
import threading
import time
import numpy as np
from gello.dynamixel.driver import DynamixelDriver


class GelloController:
    """
    GELLO hardware controller for teleoperation.
    Similar interface to VRController but uses GELLO device.
    """
    
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        joint_ids: list = None,
        joint_offsets: np.ndarray = None,
        joint_signs: np.ndarray = None,
        gripper_enabled: bool = True,
        start_joints: np.ndarray = None,
    ):
        """
        Initialize GELLO controller.
        
        Args:
            port: USB port for GELLO device
            baudrate: Communication baudrate (57600 for most GELLO)
            joint_ids: List of Dynamixel motor IDs [1,2,3,4,5,6,7,8]
            joint_offsets: Joint offset calibration (7 values for arm)
            joint_signs: Joint direction signs (7 values, ±1)
            gripper_enabled: Whether to use gripper (ID 8)
            start_joints: Initial joint configuration
        """
        # Default configuration for Franka (7 DoF + gripper)
        if joint_ids is None:
            joint_ids = [1, 2, 3, 4, 5, 6, 7, 8] if gripper_enabled else [1, 2, 3, 4, 5, 6, 7]
        
        self.joint_ids = joint_ids
        self.num_joints = 7  # Franka has 7 arm joints
        self.gripper_enabled = gripper_enabled
        
        # Calibration parameters
        if joint_offsets is None:
            # Updated offsets from calibration (2025-11-02 17:47)
            # Calibrated to: Joint4=-90°, Joint6=+90°, others=0°
            # NOTE: Re-calibrate with your specific GELLO using:
            # cd /home/robots/gello_software
            # sudo python scripts/gello_get_offset.py --port /dev/ttyUSB0 \
            #   --start-joints 0 0 0 -1.5708 0 1.5708 0 --joint-signs 1 1 1 1 1 -1 1 --gripper
            joint_offsets = np.array([1.571, 3.142, 6.283, 4.712, 3.142, 1.571, 3.142])
        if joint_signs is None:
            # Joint 6 (index 5) is inverted
            joint_signs = np.array([1, 1, 1, 1, 1, -1, 1])
            
        self.joint_offsets = np.array(joint_offsets)
        self.joint_signs = np.array(joint_signs)
        
        # Start position
        if start_joints is None:
            # Default Franka home position
            start_joints = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        self.start_joints = start_joints
        
        # Initialize Dynamixel driver
        print(f"Initializing GELLO on {port} at {baudrate} baud...")
        self.driver = DynamixelDriver(
            ids=joint_ids,
            port=port,
            baudrate=baudrate
        )
        print("✅ GELLO initialized successfully")
        
        # State
        self._current_joints = None
        self._gripper_state = 0.0  # 0=open, 1=closed
        self.reset_state()
        
    def reset_state(self):
        """Reset internal state"""
        self._current_joints = self.start_joints.copy()
        self._gripper_state = 0.0
        
    def get_joint_state(self) -> np.ndarray:
        """
        Read current GELLO joint positions.
        
        Returns:
            np.ndarray: Joint positions [j1, j2, ..., j7, gripper]
        """
        # Read raw positions from GELLO
        raw_positions = self.driver.get_joints()
        
        # Apply calibration to arm joints (first 7)
        arm_joints = np.array(raw_positions[:self.num_joints])
        calibrated_joints = (arm_joints - self.joint_offsets) * self.joint_signs
        
        # Handle gripper if enabled
        if self.gripper_enabled and len(raw_positions) > self.num_joints:
            gripper_raw = raw_positions[self.num_joints]
            # Normalize gripper to [0, 1]
            # Actual measured range: [2.542 (closed), 3.513 (open)]
            # Map: 3.513 (open) -> 0, 2.542 (closed) -> 1
            gripper_min = 2.542  # Pressed/closed
            gripper_max = 3.513  # Released/open
            gripper_normalized = np.clip((gripper_raw - gripper_min) / (gripper_max - gripper_min), 0, 1)
            gripper_normalized = 1.0 - gripper_normalized  # Invert: 0=open, 1=closed
            return np.append(calibrated_joints, gripper_normalized)
        else:
            return calibrated_joints
    
    def forward(self, state, include_info=True, method="joint_position"):
        """
        Get action from GELLO.
        Compatible with VRController interface.
        
        Args:
            state: Current robot state dict
            include_info: Whether to return additional info
            method: Control method ("joint_position" or "delta_action")
            
        Returns:
            action: Action array
            info: Additional information dict (if include_info=True)
        """
        # Read GELLO state
        gello_joints = self.get_joint_state()
        
        # Get current robot state
        if "robot_state" in state:
            if "joint_positions" in state["robot_state"]:
                current_robot_joints = state["robot_state"]["joint_positions"]
            else:
                # Fallback: assume robot is at start position
                current_robot_joints = self.start_joints
        else:
            current_robot_joints = self.start_joints
            
        if method == "delta_action":
            # Compute delta from current robot position
            delta_joints = gello_joints[:self.num_joints] - current_robot_joints[:self.num_joints]
            
            # Create action: [delta_j1, ..., delta_j7, gripper]
            if self.gripper_enabled:
                action = np.append(delta_joints, gello_joints[-1])
            else:
                action = delta_joints
                
        elif method == "joint_position":
            # Direct joint position control
            action = gello_joints
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if include_info:
            info = {
                "gello_joints": gello_joints,
                "robot_joints": current_robot_joints,
                "method": method,
            }
            return action, info
        else:
            return action
    
    def close(self):
        """Clean up resources"""
        if self.driver is not None:
            self.driver.close()
            print("GELLO driver closed")


def test_gello():
    """Simple test function"""
    print("Testing GELLO controller...")
    
    gello = GelloController(
        port="/dev/ttyUSB0",
        baudrate=57600,
    )
    
    print("\nReading GELLO positions (press Ctrl+C to stop)...")
    try:
        while True:
            joints = gello.get_joint_state()
            print(f"\rJoints: {joints[:7].round(3)}  Gripper: {joints[7]:.3f}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        gello.close()


if __name__ == "__main__":
    test_gello()
