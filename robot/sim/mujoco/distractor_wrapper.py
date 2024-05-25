import copy

import gym
import mujoco
import numpy as np

from utils.transformations_mujoco import euler_to_quat_mujoco


class DistWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        obj_id="rod",
        obj_pose_init=None,
        obj_pose_noise_dict=None,
        obj_rgba=[1., 0., 0., 1.],
        reset_data_on_reset=True,
        verbose=False,
        **kwargs
    ):
        super(DistWrapper, self).__init__(env)
        
        self.verbose = verbose

        self.dist_id = obj_id
        self.reset_data_on_reset = reset_data_on_reset

        # Mujoco object ids
        self.dist_body_id = mujoco.mj_name2id(
            self.env.unwrapped._robot.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.dist_id}_body"
        )

        self.dist_joint_id = mujoco.mj_name2id(
            self.env.unwrapped._robot.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.dist_id}_freejoint"
        )
        self.dist_joint_id = self.env.unwrapped._robot.model.jnt_qposadr[self.dist_joint_id]
            
        self.dist_geom_id = mujoco.mj_name2id(
            self.env.unwrapped._robot.model, mujoco.mjtObj.mjOBJ_GEOM, f"{self.dist_id}_geom"
        )

        # Color
        self.env.unwrapped._robot.model.geom_rgba[self.dist_geom_id] = obj_rgba

        # Object position
        self.dist_pose_noise_dict = obj_pose_noise_dict
        self.dist_pos_noise = obj_pose_noise_dict is not None
        self.init_dist_pose = self.get_dist_pose() if obj_pose_init is None else obj_pose_init
        self.curr_dist_pose = None

    def reset(self, *args, **kwargs):
        
        # # reset mujoco data -> propagate model changes to data
        if self.reset_data_on_reset:
            mujoco.mj_resetData(
                self.env.unwrapped._robot.model, self.env.unwrapped._robot.data
            )

        # randomize obj position |
        self.resample_dist_pose()

        if self.curr_dist_pose is None:
            dist_pose = self.init_dist_pose.copy()
        else:
            dist_pose = self.curr_dist_pose.copy()
        # set obj qpos | mujoco forward
        self.update_dist(dist_pose)

        # reset robot |
        obs = self.env.reset()

        return obs

    def get_dist_pose(self):
        return self.env.unwrapped._robot.data.qpos[self.dist_joint_id:self.dist_joint_id+7]
        
    def set_dist_pose(self, dist_pose):
        self.dist_pos_noise = False
        self.init_dist_pose = dist_pose.copy()
        self.update_dist(dist_pose)

    def resample_dist_pose(self):
        pose = self.init_dist_pose.copy()
        
        if self.dist_pos_noise:
            pose[0] += np.random.uniform(
                self.dist_pose_noise_dict["x"]["min"],
                self.dist_pose_noise_dict["x"]["max"],
            )
            pose[1] += np.random.uniform(
                self.dist_pose_noise_dict["y"]["min"],
                self.dist_pose_noise_dict["y"]["max"],
            )
            pose[3:7] = euler_to_quat_mujoco(
                [
                    0.0,
                    0.0,
                    np.random.uniform(
                        self.dist_pose_noise_dict["yaw"]["min"],
                        self.dist_pose_noise_dict["yaw"]["max"],
                        size=1,
                    ).item(),
                ]
            )

        if self.verbose:
            print(f"Distractor pose: {pose} - seed {self.env.unwrapped._seed}")
        self.curr_dist_pose = pose.copy()

    def update_dist(self, qpos):
        
        self.env.unwrapped._robot.data.qpos[self.dist_joint_id:self.dist_joint_id+7] = qpos
        
        # mujoco.mj_step(
        #     self.env.unwrapped._robot.model,
        #     self.env.unwrapped._robot.data,
        #     nstep=self.env.unwrapped._robot.frame_skip,
        # )
        