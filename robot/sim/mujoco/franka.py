import os

import torch

from robot.franka_base import FrankaBase

os.environ["MUJOCO_GL"] = "egl"

import copy

import mujoco
import mujoco_viewer
import numpy as np

from helpers.transformations import (euler_to_quat, quat_to_euler,
                                     rmat_to_euler, rmat_to_quat)
from robot.controllers.utils import generate_joint_space_min_jerk


class FrankaMujoco(FrankaBase):
    def __init__(
        self,
        # robot
        robot_type="panda",
        control_hz=15,
        gripper=True,
        custom_controller=True,
        gain_scale=1.5,
        reset_gain_scale=1.0,
        # rendering
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["front", "left"],
        use_rgb=True,
        use_depth=True,
        img_height=480,
        img_width=640,
        # noise
        obs_noise=0.0,
        act_drop=0.0,
        # mujoco
        seed=0,
        xml_path="robot/sim/mujoco/assets/base_franka.xml",
    ):
        
        super().__init__(robot_type=robot_type, control_hz=control_hz, gripper=gripper, custom_controller=custom_controller)
        self.policy = None
        self.gain_scale = gain_scale
        self.reset_gain_scale = reset_gain_scale

        # rendering
        assert (
            has_renderer and has_offscreen_renderer
        ) is False, "both has_renderer and has_offscreen_renderer not supported"
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.viewer = None
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.img_width = img_width
        self.img_height = img_height
        self.camera_names = camera_names

        self.obs_noise = obs_noise
        self.act_drop = act_drop

        # setup model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model_backup = copy.deepcopy(self.model)
        self.data = mujoco.MjData(self.model)

        # TODO verify this
        self.frame_skip = int((1/self.control_hz) / self.model.opt.timestep)

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "EEF")
        self.franka_joint_ids = []
        for i in range(7):
            self.franka_joint_ids += [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot:joint{i+1}"
                )
            ]

        self.franka_body_names = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"]

        # polymetis control: load robot model
        import torchcontrol as toco
        robot_description_path = "robot/real/config/panda_arm.urdf"
        ee_link_name = "panda_link8"
        self.robot_model = toco.models.RobotModelPinocchio(
                    robot_description_path, ee_link_name
                )

        # polymetis control: load robot hardware config
        import yaml

        # polymetis/polymetis/conf/robot_client/franka_hardware.yaml -> replaced by R2D2 
        with open("robot/real/config/franka_hardware.yaml", "r") as stream:
            try:
                franka_hardware_conf = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.metadata = franka_hardware_conf["robot_client"]["metadata_cfg"]

        self.seed(seed)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def get_ee_pose(self):
        return self.get_ee_pos(), self.get_ee_angle()
    
    def get_ee_pos(self):
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        if self.obs_noise:
            ee_pos = self.apply_obs_noise(ee_pos)
        return ee_pos
    
    def get_ee_angle(self, quat=False):
        ee_mat = self.data.site_xmat[self.ee_site_id].copy().reshape(3, 3)
        ee_angle = rmat_to_euler(ee_mat)
        if quat:
            return euler_to_quat(ee_angle)
        else:
            return ee_angle
        
    def get_joint_positions(self):
        qpos = self.data.qpos[self.franka_joint_ids].copy()
        if self.obs_noise:
            qpos = self.apply_obs_noise(qpos)
        return qpos

    def get_joint_velocities(self):
        qvel = self.data.qvel[self.franka_joint_ids].copy()
        if self.obs_noise:
            qvel = self.apply_obs_noise(qvel)
        return qvel

    def get_robot_state(self):

        joint_positions = self.get_joint_positions()
        joint_velocities = self.get_joint_velocities()

        gripper_position = self.get_gripper_state()
        pos, quat = self.get_ee_pos(), self.get_ee_angle(quat=True)

        state_dict = {
            "cartesian_position": np.concatenate([pos, quat]),
            "gripper_position": gripper_position,
            "joint_positions": torch.tensor(joint_positions, dtype=torch.float32),
            "joint_velocities": torch.tensor(joint_velocities, dtype=torch.float32),
            # "joint_torques_computed": list(robot_state.joint_torques_computed),
            # "prev_joint_torques_computed": list(
            #     robot_state.prev_joint_torques_computed
            # ),
            # "prev_joint_torques_computed_safened": list(
            #     robot_state.prev_joint_torques_computed_safened
            # ),
            # "motor_torques_measured": list(robot_state.motor_torques_measured),
            # "prev_controller_latency_ms": robot_state.prev_controller_latency_ms,
            # "prev_command_successful": robot_state.prev_command_successful,
        }

        # timestamp_dict = {
        #     "robot_timestamp_seconds": robot_state.timestamp.seconds,
        #     "robot_timestamp_nanos": robot_state.timestamp.nanos,
        # }
        timestamp_dict = {}

        return state_dict, timestamp_dict
    
    def get_gripper_state(self):
        return np.array(0)

    def update_pose(self, pos, angle, gripper=None):
        pos, angle = self.apply_action_drop(pos, angle)
        desired_qpos, success = self.ik.compute(pos, angle)

        self.data.ctrl[: len(desired_qpos)] = desired_qpos

        if gripper is not None:
        # TODO gripper in sim
            self.data.ctrl[-1] = gripper * 255
            
        # advance simulation, use control callback to obtain external force and control.
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        if self.has_renderer:
            self.render()

        return self.get_ee_pos(), self.get_ee_angle()

    def is_running_policy(self):
        return self.policy is not None
    
    def _start_custom_controller(self, q_desired=None, ctrl_mode=0.0):
        from robot.controllers.mixed_cartesian_impedance import \
            MixedCartesianImpedanceControl

        self.policy = MixedCartesianImpedanceControl(
            joint_pos_current=torch.tensor(self.get_joint_positions()),
            Kp=self.gain_scale
            * torch.tensor(
                self.metadata["default_Kx"], dtype=torch.float32
            ),  # the higher, the faster it returns to pos
            Kd=self.gain_scale
            * torch.tensor(
                self.metadata["default_Kxd"], dtype=torch.float32
            ),  # the higher, the stiffer around pos
            kp_pos=self.gain_scale * torch.tensor(self.metadata["default_Kq"], dtype=torch.float32),
            kd_pos=self.gain_scale * torch.tensor(self.metadata["default_Kqd"], dtype=torch.float32),
            desired_joint_pos=torch.tensor(self.get_joint_positions(), dtype=torch.float32) if q_desired is None else q_desired.float(),
            ctrl_mode=ctrl_mode,
            robot_model=self.robot_model,
            ignore_gravity=True,
        )
        return self.policy

    def _terminate_current_controller(self):
        self.policy = None

    def _update_current_controller(self, udpate_pkt):
        self._start_custom_controller(q_desired=udpate_pkt["q_desired"] if "q_desired" in udpate_pkt else None, ctrl_mode=udpate_pkt["ctrl_mode"])
        return self.policy.forward(self.get_robot_state()[0])
    
    # def update_joints(self, qpos, velocity=False, blocking=False):
        # self.data.qpos[: len(qpos)] = qpos
        # # forward dynamics: same as mj_step but do not integrate in time.
        # mujoco.mj_forward(self.model, self.data)

    def add_noise_to_joints(self, original_joints, cartesian_noise):
        original_joints = torch.Tensor(original_joints)

        pos, quat = self._robot.get_ee_pos(), self._robot.get_ee_angle(quat=True)
        
        curr_pose = pos.tolist() + quat_to_euler(quat).tolist()
        new_pose = add_poses(cartesian_noise, curr_pose)

        new_pos = torch.Tensor(new_pose[:3])
        new_quat = torch.Tensor(euler_to_quat(new_pose[3:]))

        noisy_joints, success = self._robot.solve_inverse_kinematics(
            new_pos, new_quat, original_joints
        )

        if success:
            desired_joints = noisy_joints
        else:
            desired_joints = original_joints

        return desired_joints.tolist()
    
    def _adaptive_time_to_go_polymetis(self, robot_model, joint_displacement: torch.Tensor, time_to_go_default=1.):
        """Compute adaptive time_to_go
        Computes the corresponding time_to_go such that the mean velocity is equal to one-eighth
        of the joint velocity limit:
        time_to_go = max_i(joint_displacement[i] / (joint_velocity_limit[i] / 8))
        (Note 1: The magic number 8 is deemed reasonable from hardware tests on a Franka Emika.)
        (Note 2: In a min-jerk trajectory, maximum velocity is equal to 1.875 * mean velocity.)

        The resulting time_to_go is also clipped to a minimum value of the default time_to_go.
        """
        joint_vel_limits = robot_model.get_joint_velocity_limits()
        joint_pos_diff = torch.abs(joint_displacement)
        time_to_go = torch.max(joint_pos_diff / joint_vel_limits * 8.0)
        return max(time_to_go, time_to_go_default)
    
    def adaptive_time_to_go(self, desired_joint_position, t_min=1, t_max=4):
        curr_joint_position = self.get_joint_positions()
        displacement = desired_joint_position - curr_joint_position
        time_to_go = self._adaptive_time_to_go_polymetis(self.robot_model, displacement)
        clamped_time_to_go = min(t_max, max(time_to_go, t_min))
        return clamped_time_to_go
    
    def update_joints(
        self, command, velocity=False, blocking=False, cartesian_noise=None
    ):
        if cartesian_noise is not None:
            command = self.add_noise_to_joints(command, cartesian_noise)
        command = torch.Tensor(command)

        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self.get_joint_positions()

        # BLOCKING EXECUTION
        # make sure custom controller is running
        if blocking and self.custom_controller:
            if not self.is_running_policy():
                self._start_custom_controller()
            time_to_go = self.adaptive_time_to_go(command)
            self.move_to_joint_positions(command, time_to_go=time_to_go)
        # kill cartesian impedance
        elif blocking:
            if self.is_running_policy():
                self._terminate_current_controller()

            time_to_go = self.adaptive_time_to_go(command)
            self._robot.move_to_joint_positions(command, time_to_go=time_to_go)

            self._robot.start_cartesian_impedance()

        # NON BLOCKING
        else:

            def helper_non_blocking():
                if not self.is_running_policy():
                    if self.custom_controller:
                        self._start_custom_controller()
                    else:
                        self._robot.start_cartesian_impedance()
                
                if self.custom_controller:
                    self.update_desired_joint_positions(command)
                else:
                    self._robot.update_desired_joint_positions(command)

            # run_threaded_command(helper_non_blocking)
            helper_non_blocking()

    # def update_gripper(self, gripper, velocity=False, blocking=False):
    #     # TODO gripper in sim
    #     self.data.ctrl[-1] = gripper * 255
    #     mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def update_desired_joint_positions(self, q_desired=None, kp=None, kd=None):
        """update joint pos"""
        udpate_pkt = {}
        udpate_pkt["ctrl_mode"] = 0.0 # torch.Tensor([0.0])

        if q_desired is not None:
            udpate_pkt["q_desired"] = (
                q_desired if torch.is_tensor(q_desired) else torch.tensor(q_desired)
            )
        if kp is not None:
            udpate_pkt["kp"] = kp if torch.is_tensor(kp) else torch.tensor(kp)
        if kd is not None:
            udpate_pkt["kd"] = kd if torch.is_tensor(kd) else torch.tensor(kd)
        assert udpate_pkt, "Atleast one parameter needs to be specified for udpate"

        output_pkt = self._update_current_controller(udpate_pkt)

        # TODO torque control
        self.data.ctrl[: len(self.franka_joint_ids)] = output_pkt["joint_torques"].detach().numpy() + self.data.qfrc_bias
        mujoco.mj_step(self.model, self.data)
        # mujoco.mj_step(self.model, self.data, nstep=self.frame_skip // 10)
        
        # if "q_desired" in udpate_pkt:
        #     self.data.qpos[: len(self.franka_joint_ids)] = udpate_pkt["q_desired"]
        #     mujoco.mj_forward(self.model, self.data)

    def move_to_joint_positions(self, q_desired=None, time_to_go=3):
        # Use registered controller
        q_current = torch.tensor(self.get_joint_positions())

        # generate min jerk trajectory
        dt = 0.1
        waypoints = generate_joint_space_min_jerk(
            start=q_current, goal=q_desired, time_to_go=time_to_go, dt=dt
        )
        # reset using min_jerk traj
        for i in range(len(waypoints)):
            self.update_desired_joint_positions(
                q_desired=waypoints[i]["position"],
                kp=self.reset_gain_scale
                * torch.tensor(self.metadata["default_Kq"], dtype=torch.float32),
                kd=self.reset_gain_scale
                * torch.tensor(self.metadata["default_Kqd"], dtype=torch.float32),
            )
            # time.sleep(dt)

        # reset back gains to gain-policy
        self.update_desired_joint_positions(
            kp=self.gain_scale * torch.tensor(self.metadata["default_Kq"], dtype=torch.float32),
            kd=self.gain_scale * torch.tensor(self.metadata["default_Kqd"], dtype=torch.float32),
        )

    def update_gripper(self, command, velocity=True, blocking=False):
        
        # TODO grasping for sim
        return 
    
        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            command = gripper_delta + self.get_gripper_position()

        command = float(np.clip(command, 0, 1))
        # https://github.com/facebookresearch/fairo/issues/1398
        # for robotiq consider using
        # self._gripper.grasp(grasp_width=self._max_gripper_width * (1 - command), speed=0.05, force=0.5, blocking=blocking)
        # for franka gripper, use discrete grasp/ungrasp
        if command > 0.0:
            self._gripper.grasp(
                grasp_width=0.0, speed=0.5, force=5.0, blocking=blocking
            )
        else:
            self._gripper.grasp(
                grasp_width=self._max_gripper_width,
                speed=0.5,
                force=5.0,
                blocking=blocking,
            )

    def apply_action_drop(self, pos, angle):
        # with act_drop% chance use last action to simulate delay on real system
        if self.act_drop > 0. and np.random.choice(
            [0, 1], p=[1 - self.act_drop, self.act_drop]
        ):
            return self._last_pos, self._last_angle
        self._last_pos = pos.copy()
        self._last_angle = angle.copy()
        return self._last_pos, self._last_angle

    def apply_obs_noise(self, obs):
        if self.noiseless:
            return obs
        if self.obs_noise is not None:
            obs += np.random.normal(loc=0.0, scale=self.obs_noise, size=obs.shape)
        return obs

    def viewer_setup(self):
        if self.has_renderer:
            return mujoco_viewer.MujocoViewer(
                self.model, self.data, height=720, width=720
            )
        if self.has_offscreen_renderer:
            return mujoco.Renderer(
                self.model, width=self.img_width, height=self.img_height
            )

    def render(self, mode="rgb_array"):
        assert mode in ["rgb_array", "human"], "mode not in ['rgb_array', 'human']"
        assert (
            self.has_renderer or self.has_offscreen_renderer
        ), "no renderer available."

        imgs = []

        if not self.viewer:
            self.viewer = self.viewer_setup()

        if self.has_renderer and mode == "human":
            self.viewer.render()
        elif self.has_offscreen_renderer and mode == "rgb_array":
            for camera in self.camera_names:
                self.viewer.update_scene(self.data, camera=camera)
                if self.use_depth:
                    self.viewer.enable_depth_rendering()
                    depth = self.viewer.render().copy()
                    self.viewer.disable_depth_rendering()
                
                if not self.use_rgb and self.use_depth:
                    color_image = np.zeros((depth.shape[0], depth.shape[1], 3))
                else:
                    color_image = self.viewer.render().copy()
                
                dict_1 = {'serial_number': camera, 'array': color_image, 'shape': color_image.shape, 'type': 'rgb'}
                dict_2 = {'serial_number': camera, 'array': depth, 'shape': depth.shape, 'type': 'depth'}
                imgs.append(dict_1)
                imgs.append(dict_2)

        return imgs

    @property
    def parameter_dim(self):
        return len(self.get_parameters())

    def seed(self, seed=0):
        np.random.seed(seed)

    def geoms_from_body_names(self, names):
        geom_ids = []
        for name in names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) 
            geom_ids += [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == body_id]
        return geom_ids
    
    def is_geom_contact(self, geom_list1, geom_list2):
        contacts_from_1_to_2 = []
        contacts_from_2_to_1 = []
        for contact in self.data.contact:
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            if geom1_id in geom_list1 and geom2_id in geom_list2:
                contacts_from_1_to_2.append((geom1_id, geom2_id))
            elif geom1_id in geom_list2 and geom2_id in geom_list1:
                contacts_from_2_to_1.append((geom1_id, geom2_id))
        return len(contacts_from_1_to_2) > 0 or len(contacts_from_2_to_1) > 0
    
    def is_franka_contact(self):
        franka_geom_ids = self.geoms_from_body_names(self.franka_body_names)
        obstacle_geom_ids = self.geoms_from_body_names(self.obstacles_body_names)
        return self.is_geom_contact(franka_geom_ids, obstacle_geom_ids)
    