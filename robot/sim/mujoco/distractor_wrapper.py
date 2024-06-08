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

        mujoco.mj_resetData(
            self.env.unwrapped._robot.model, self.env.unwrapped._robot.data
        )

    def reset(self, *args, **kwargs):
        
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

    def sample_within_constraints(self,existing_pos, size_k, x_range, y_range):
        x1, y1 = existing_pos
        min_dist = 3 * size_k

        # valid = False
        # while not valid:
        
        x2, y2 = x1, y1
        for _ in range(100):
            
            # Randomly choose an angle and distance
            x_sign = np.random.choice([-1, 1])
            y_sign = np.random.choice([-1, 1])
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(min_dist, min_dist + 0.1)  # Adjust maximum as needed

            # Calculate potential new position
            x2 = x1 + distance * np.cos(angle) * x_sign
            y2 = y1 + distance * np.sin(angle) * y_sign

            # Check if the new position is within the specified x and y ranges
            # if x_range[0] <= x2 <= x_range[1] and y_range[0] <= y2 <= y_range[1]:
            #     valid = True
        
        return x2, y2

    def resample_dist_pose(self):
        pose = self.init_dist_pose.copy()
      
        main_object_pose = self.env.get_obj_pose()[:2]
      
        if self.dist_pos_noise:
            # pose[0] += np.random.uniform(
            #     self.dist_pose_noise_dict["x"]["min"],
            #     self.dist_pose_noise_dict["x"]["max"],
            # )
            # pose[1] += np.random.uniform(
            #     self.dist_pose_noise_dict["y"]["min"],
            #     self.dist_pose_noise_dict["y"]["max"],
            # )
            x_range = [pose[0]+self.dist_pose_noise_dict["x"]["min"],pose[0]+self.dist_pose_noise_dict["x"]["max"]]
            y_range = [pose[1]+self.dist_pose_noise_dict["y"]["min"],pose[1]+self.dist_pose_noise_dict["y"]["max"]]
            pose[0], pose[1] = self.sample_within_constraints(main_object_pose,size_k=0.05,x_range=x_range,y_range=y_range)

            pose[3:7] = euler_to_quat_mujoco([
                0.0,
                0.0,
                np.random.uniform(
                    self.dist_pose_noise_dict["yaw"]["min"],
                    self.dist_pose_noise_dict["yaw"]["max"],
                    size=1,
                ).item(),
            ])

        if self.verbose:
            print(f"Distractor pose: {self.env.get_obj_pose()} - seed {self.env.unwrapped._seed}")
        self.curr_dist_pose = pose.copy()
        return pose

    def update_dist(self, qpos):
        
        self.env.unwrapped._robot.data.qpos[self.dist_joint_id:self.dist_joint_id+7] = qpos
        
        mujoco.mj_step(
            self.env.unwrapped._robot.model,
            self.env.unwrapped._robot.data,
            nstep=self.env.unwrapped._robot.frame_skip,
        )
        