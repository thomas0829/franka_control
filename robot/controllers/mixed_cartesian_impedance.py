from typing import Dict

import torch
import torchcontrol as toco
from torchcontrol.utils import to_tensor
from torchcontrol.utils.tensor_utils import diagonalize_gain, to_tensor


class MixedCartesianImpedanceControl(toco.PolicyModule):
    """
    Performs impedance control in Cartesian space.
    Errors and feedback are computed in Cartesian space, and the resulting forces are projected back into joint space.
    """

    def __init__(
        self,
        joint_pos_current,
        Kp,
        Kd,
        kp_pos,
        kd_pos,
        desired_joint_pos,
        robot_model: torch.nn.Module,
        ctrl_mode=1,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in Cartesian space
            Kd: D gains in Cartesian space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        self.kp = torch.nn.Parameter(kp_pos)
        self.kd = torch.nn.Parameter(kd_pos)
        self.q_desired = torch.nn.Parameter(desired_joint_pos)
        self.feedback = toco.modules.JointSpacePD(self.kp, self.kd)

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.pose_pd = toco.modules.feedback.CartesianSpacePDFast(Kp, Kd)

        self.joint_pd = toco.modules.feedback.JointSpacePD(self.kp, self.kd)

        # Reference pose
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

        # Reference pose
        joint_pos_current = to_tensor(joint_pos_current)
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
        self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)
        self.ee_vel_desired = torch.nn.Parameter(torch.zeros(3))
        self.ee_rvel_desired = torch.nn.Parameter(torch.zeros(3))
        self.ctrl_mode = torch.nn.Parameter(
            torch.ones(1) * ctrl_mode
        )  # Mode 0 joint PD Mode 1 cartesian

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        if self.ctrl_mode.data == 1:

            joint_pos_current = state_dict["joint_positions"]
            joint_vel_current = state_dict["joint_velocities"]

            self.ee_pos_desired = torch.nn.Parameter(state_dict["ee_pos_desired"])
            self.ee_quat_desired = torch.nn.Parameter(state_dict["ee_quat_desired"])

            # Control logic
            ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
                joint_pos_current
            )
            jacobian = self.robot_model.compute_jacobian(joint_pos_current)
            ee_twist_current = jacobian @ joint_vel_current

            wrench_feedback = self.pose_pd(
                ee_pos_current,
                ee_quat_current,
                ee_twist_current,
                self.ee_pos_desired,
                self.ee_quat_desired,
                torch.cat([self.ee_vel_desired, self.ee_rvel_desired]),
            )
            torque_feedback = jacobian.T @ wrench_feedback

            torque_feedforward = self.invdyn(
                joint_pos_current,
                joint_vel_current,
                torch.zeros_like(joint_pos_current),
            )  # coriolis

            torque_out = torque_feedback + torque_feedforward

            print(torque_out)
            return {"joint_torques": torque_out}
        else:

            # PD impedance control

            # State extraction
            joint_pos_current = state_dict["joint_positions"]
            joint_vel_current = state_dict["joint_velocities"]

            # # temp fix
            self.joint_pos_desired = torch.nn.Parameter(to_tensor(self.q_desired))
            self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

            # Control logic
            torque_feedback = self.joint_pd(
                joint_pos_current,
                joint_vel_current,
                self.joint_pos_desired,
                self.joint_vel_desired,
            )
            torque_feedforward = self.invdyn(
                joint_pos_current,
                joint_vel_current,
                torch.zeros_like(joint_pos_current),
            )  # coriolis
            torque_out = torque_feedback + torque_feedforward

            return {"joint_torques": torque_out}
