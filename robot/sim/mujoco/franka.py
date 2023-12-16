from robot.franka_base import FrankaBase

import os

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
from robot.real.inverse_kinematics.robot_ik_solver import RobotIKSolver
import mujoco_viewer
import copy

from helpers.transformations import euler_to_quat, quat_to_euler, rmat_to_quat, rmat_to_euler


class FrankaMujoco(FrankaBase):
    def __init__(
        self,
        # rendering
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["front", "left"],
        use_rgb=True,
        use_depth=True,
        img_height=480,
        img_width=640,
        # 
        control_hz=10,
        # noise
        obs_noise=0.0,
        act_drop=0.0,
        # mujoco
        seed=0,
        xml_path="robot/sim/mujoco/assets/base_franka.xml",
    ):
        
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
        self.frame_skip = int((1/control_hz) / self.model.opt.timestep)

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "EEF")
        self.franka_joint_ids = []
        for i in range(7):
            self.franka_joint_ids += [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot:joint{i+1}"
                )
            ]

        self.franka_body_names = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"]
        
        # setup IK
        self.ik = RobotIKSolver(robot=self, control_hz=control_hz)

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

    def update_joints(self, qpos):
        self.data.qpos[: len(qpos)] = qpos
        # forward dynamics: same as mj_step but do not integrate in time.
        mujoco.mj_forward(self.model, self.data)

    def update_gripper(self, gripper):
        # TODO gripper in sim
        self.data.ctrl[-1] = gripper * 255
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

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
                
                dict_1 = {'array': color_image, 'shape': color_image.shape, 'type': 'rgb'}
                dict_2 = {'array': depth, 'shape': depth.shape, 'type': 'depth'}
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
    