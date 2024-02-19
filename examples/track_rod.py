import time

import matplotlib.pyplot as plt
import numpy as np

from perception.trackers.color_tracker import ColorTracker
from robot.robot_env import RobotEnv
from utils.pointclouds import crop_points

if __name__ == "__main__":

    control_hz = 10

    env = RobotEnv(
        control_hz=control_hz,
        DoF=6,
        robot_type="fr3",
        ip_address="172.16.0.1",
        camera_model="realsense",
        max_lin_vel=1.0,
        max_rot_vel=1.0,
        max_path_length=100,
    )

    tracker = ColorTracker(outlier_removal=True)

    # define workspace
    crop_min = [0.0, -0.4, -0.1]
    crop_max = [0.5, 0.4, 0.5]

    rod_poses_raw = []
    rod_poses_filter = []
    for i in range(50):

        # prepare obs
        obs_dict = env.get_images_and_points()
        rgbs, points = [], []
        for key in obs_dict.keys():
            rgbs.append(obs_dict[key]["rgb"])
            points.append(obs_dict[key]["points"])

        # track points
        tracked_points = tracker.track_multiview(rgbs, points, color="red", show=False)
        # crop to workspace
        cropped_points = crop_points(
            tracked_points, crop_min=crop_min, crop_max=crop_max
        )

        # compare raw and filtered rod pose
        rod_poses_raw.append(
            tracker.get_rod_pose(cropped_points, lowpass_filter=False, show=False)
        )
        rod_poses_filter.append(
            tracker.get_rod_pose(
                cropped_points,
                lowpass_filter=True,
                cutoff_freq=1,
                control_hz=control_hz,
                show=False,
            )
        )

        time.sleep(1 / control_hz)

    plt.plot(
        np.stack(rod_poses_raw)[:, 0],
        np.stack(rod_poses_raw)[:, 1],
        label="rod pose raw",
    )
    plt.plot(
        np.stack(rod_poses_filter)[:, 0],
        np.stack(rod_poses_filter)[:, 1],
        label="rod pose filter",
    )

    plt.legend()
    plt.show()
