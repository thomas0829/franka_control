import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from utils.pointclouds import crop_points


class ColorTracker:
    def __init__(self, outlier_removal=True):
        self.outlier_removal = outlier_removal
        self.reset()

    def reset(self):
        self._tracker_history = None

    def show_mask(self, rgb, mask):
        gray_img = np.mean(rgb, axis=2) / 255.0
        rgb_gray = np.stack([gray_img] * 3, axis=2)

        colored_mask = np.zeros_like(rgb_gray)
        colored_mask[:, :, 0] = 255.0

        masked_image = np.where(mask[:, :, None] > 0, colored_mask, rgb_gray)

        plt.imshow(masked_image)
        plt.show()

    def get_mask(self, rgb, color):
        template_mask = self.get_color_mask(rgb.copy(), color=color)

        kernel = np.ones((3, 3), np.uint8)
        template_mask = cv2.erode(template_mask, kernel)
        template_mask = cv2.dilate(template_mask, kernel)

        return template_mask

    def track(self, rgb, points, color, show=False):
        # generate color mask
        mask = self.get_mask(rgb, color)
        if show:
            self.show_mask(rgb, mask)
        # crop points by mask
        points = points[mask.reshape(-1) > 0]

        if self.outlier_removal:
            points = self.remove_outliers(points)

        return points

    def track_multiview(self, rgbs, points, color, show=False):
        for i, (rgb, point) in enumerate(zip(rgbs, points)):
            if i == 0:
                points = self.track(rgb, point, color, show=show)
            else:
                points = np.concatenate(
                    (points, self.track(rgb, point, color, show=show)), axis=0
                )

        if self.outlier_removal:
            points = self.remove_outliers(points)

        return points

    def remove_outliers(self, points):
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(points)
        o3d_pcd, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)
        points = np.asarray(o3d_pcd.points)
        return points

    def get_rod_pose(
        self, points, lowpass_filter=False, cutoff_freq=1, control_hz=10, show=False
    ):
        from utils.pointclouds import points_to_pcd, visualize_pcds
        from utils.transformations import euler_to_quat

        # crop to height
        points_crop = crop_points(
            points, crop_min=[-1.0, -1.0, 0.01], crop_max=[1.0, 1.0, 0.05]
        )

        # fit cuboid
        # plano1 = pyrsc.Cuboid()
        # best_eq, best_inliers = plano1.fit(points_crop, thresh=0.05, maxIteration=500)

        rod_pcd = points_to_pcd(points_crop)
        # plane = rod_pcd.select_by_index(best_inliers).paint_uniform_color([1, 0, 0])

        # Fit the best oriented bounding box to the cuboid points.
        # obb = plane.get_oriented_bounding_box()

        # if show:
        #     obb.color = np.array([255.0, 0.0, 0])

        #     mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.5, depth=1e-3)
        #     mesh_box = mesh_box.translate(np.array([-0.75, -0.75, 5e-4]))

        #     visualize_pcds([obb, rod_pcd, mesh_box])

        # center = obb.center.copy()

        # get most outer points
        points_max = points_crop.argmax(axis=0)
        points_min = points_crop.argmin(axis=0)
        boxpoints = np.concatenate(
            (points_crop[points_max], points_crop[points_min]), axis=0
        )
        # collapse to (almost) 2D plane
        boxpoints[:, 2] /= 100
        box = points_to_pcd(boxpoints)

        # fit the best oriented bounding box to the "plane".
        obb = box.get_oriented_bounding_box()
        obb.color = np.array([255.0, 0.0, 0])

        if show:
            mesh_box = o3d.geometry.TriangleMesh.create_box(
                width=1.5, height=1.5, depth=1e-3
            )
            mesh_box = mesh_box.translate(np.array([-0.75, -0.75, 5e-4]))

            visualize_pcds([obb, box, rod_pcd, mesh_box])

        center = obb.center.copy()

        # compute orientation
        left_idxs = np.asarray(boxpoints[:, 1] > obb.center[1])
        right_idxs = np.asarray(boxpoints[:, 1] <= obb.center[1])
        left_val = boxpoints[left_idxs].mean(axis=0)
        right_val = boxpoints[right_idxs].mean(axis=0)

        deg_z = np.arctan2(right_val[0] - left_val[0], left_val[1] - right_val[1])
        degs = np.asarray([0, 0, deg_z])
        quats = euler_to_quat(degs)

        rod_pose = np.concatenate((center, quats))

        if lowpass_filter:
            if self._tracker_history is None:
                self._tracker_history = rod_pose[None]
                return rod_pose
            else:
                self._tracker_history = np.concatenate(
                    (self._tracker_history, rod_pose[None]), axis=0
                )

                for i in range(len(rod_pose)):
                    rod_pose[i] = self.lowpass_filter(
                        self._tracker_history[:, i], cutoff_freq, control_hz
                    )[-1]

        return rod_pose

    def get_color_mask(self, rgb, color="red"):
        init_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        if color == "red":
            lower1 = np.array([0, 100, 20])
            upper1 = np.array([5, 255, 255])
            lower2 = np.array([165, 100, 20])
            upper2 = np.array([179, 255, 255])
            lower_mask = cv2.inRange(init_hsv, lower1, upper1)
            upper_mask = cv2.inRange(init_hsv, lower2, upper2)
            full_mask = lower_mask + upper_mask
        elif color == "yellow":
            # lower = np.array([20, 100, 20])
            # upper = np.array([30, 255, 255])
            lower = np.array([20, 100, 60])
            upper = np.array([25, 255, 255])
            full_mask = cv2.inRange(init_hsv, lower, upper)
        elif color == "tennis":
            lower = np.array([30, 80, 50])
            upper = np.array([55, 225, 145])
            full_mask = cv2.inRange(init_hsv, lower, upper)
        elif color == "green":
            lower = np.array([45, 100, 20])
            upper = np.array([75, 255, 255])
            full_mask = cv2.inRange(init_hsv, lower, upper)
        elif color == "blue":
            lower = np.array([100, 100, 20])
            upper = np.array([140, 255, 255])
            full_mask = cv2.inRange(init_hsv, lower, upper)
        elif color == "bouncy_green":
            lower = np.array([45, 100, 40])
            upper = np.array([75, 255, 255])
            full_mask = cv2.inRange(init_hsv, lower, upper)
        elif color == "bouncy_orange":
            lower1 = np.array([0, 100, 20])
            upper1 = np.array([5, 255, 255])
            lower2 = np.array([165, 100, 20])
            upper2 = np.array([179, 255, 255])
            lower_mask = cv2.inRange(init_hsv, lower1, upper1)
            upper_mask = cv2.inRange(init_hsv, lower2, upper2)
            full_mask = lower_mask + upper_mask

        return full_mask

    def lowpass_filter(self, signal, cutoff_freq, sample_rate):
        # Generate the Fourier Transform of the signal
        fourier = np.fft.fft(signal)
        # Generate frequencies associated with Fourier Transform
        frequencies = np.fft.fftfreq(len(fourier), 1 / sample_rate)
        # Zero out frequencies above the cutoff
        fourier[np.abs(frequencies) > cutoff_freq] = 0
        # Perform inverse Fourier Transform to get back to time domain
        filtered_signal = np.fft.ifft(fourier)
        return np.real(filtered_signal)
