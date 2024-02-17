# https://github.com/ahmeda14960/iris_robots/blob/main/camera_utils/zed_camera.py

import cv2
import numpy as np
import time
from copy import deepcopy

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("WARNING: You have not setup the ZED cameras, and currently cannot use them")


def time_ms():
    return time.time_ns() // 1_000_000


def gather_zed_cameras():
    all_zed_cameras = []
    try:
        cameras = sl.Camera.get_device_list()
    except NameError:
        return []

    for cam in cameras:
        cam = ZedCamera(cam)
        all_zed_cameras.append(cam)

    return all_zed_cameras


class ZedCamera:
	def __init__(self, camera):
		self._serial_number = str(camera.serial_number)
		self._current_params = dict(
			# depth_mode=sl.DEPTH_MODE.NEURAL,
			# coordinate_units=sl.UNIT.METER, # force milimeters
			depth_minimum_distance=0.1,
			# depth_stabilization=False,
			camera_resolution=sl.RESOLUTION.HD720,
			camera_fps=30,
			camera_image_flip=sl.FLIP_MODE.OFF,
		)
		# https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view
		# vertical FOV for a ZED 2 and sl.RESOLUTION.HD720
		self._fovy = 68

		self._extriniscs = {}

		self.depth = True
		self.pointcloud = True

		self.crop = True

		self._configure_camera()

	def _configure_camera(self):
		# Close Existing Camera #
		self.disable_camera()

		# Initialize Readers #
		self._cam = sl.Camera()
		self._sbs_img = sl.Mat()
		self._left_img = sl.Mat()
		self._right_img = sl.Mat()
		self._left_depth = sl.Mat()
		self._right_depth = sl.Mat()
		self._left_pointcloud = sl.Mat()
		self._right_pointcloud = sl.Mat()
		self._runtime = sl.RuntimeParameters()

		sl_params = sl.InitParameters(**self._current_params)
		sl_params.set_from_serial_number(int(self._serial_number))
		status = self._cam.open(sl_params)
		# self._cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)
		if status != sl.ERROR_CODE.SUCCESS:
			raise RuntimeError("Camera Failed To Open")

		self.latency = int(2.5 * (1e3 / sl_params.camera_fps))
		calib_params = (
			self._cam.get_camera_information().camera_configuration.calibration_parameters
		)
		self._intrinsics = {
			self._serial_number
			+ "_left": self._process_intrinsics(calib_params.left_cam),
			self._serial_number
			+ "_right": self._process_intrinsics(calib_params.right_cam),
		}

	def _process_intrinsics(self, params):
		intrinsics = {}
		intrinsics["cameraMatrix"] = np.array(
			[[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]]
		)
		intrinsics["distCoeffs"] = np.array(list(params.disto))
		return intrinsics

	def get_intrinsics(self):
		return deepcopy(self._intrinsics)

	def read_camera(self):
		# Read Camera #
		timestamp_dict = {self._serial_number + "_read_start": time_ms()}
		err = self._cam.grab(self._runtime)
		if err != sl.ERROR_CODE.SUCCESS:
			return None
		timestamp_dict[self._serial_number + "_read_end"] = time_ms()

		# Benchmark Latency #
		received_time = self._cam.get_timestamp(
			sl.TIME_REFERENCE.IMAGE
		).get_milliseconds()
		timestamp_dict[self._serial_number + "_frame_received"] = received_time
		timestamp_dict[self._serial_number + "_estimated_capture"] = (
			received_time - self.latency
		)

		self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT)

		left_img = deepcopy(self._left_img.get_data())
		left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

		if self.crop:
			left_img = left_img[:,355:-205]

		dict_1 = {
			"array": left_img,
			"shape": left_img.shape,
			"type": "rgb",
			"read_time": received_time,
			"serial_number": self._serial_number + "/left",
		}

		# self._cam.retrieve_image(self._right_img, sl.VIEW.RIGHT)

		# right_img = deepcopy(self._right_img.get_data())
		# right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

		# dict_1 = {'array': right_img,  'shape': right_img.shape, 'type': 'rgb',
		# 	'read_time': received_time, 'serial_number': self._serial_number + '/right'}

		if self.depth:
			self._cam.retrieve_measure(self._left_depth, sl.MEASURE.DEPTH)

			left_depth = deepcopy(self._left_depth.get_data())

			if self.crop:
				left_depth = left_depth[:,355:-205]

			dict_2 = {
				"array": left_depth,
				"shape": left_depth.shape,
				"type": "depth",
				"read_time": received_time,
				"serial_number": self._serial_number + "/left",
			}

			# self._cam.retrieve_measure(self._right_depth, sl.MEASURE.DEPTH_RIGHT)
			
			# right_depth = deepcopy(self._right_depth.get_data())

			# dict_2 = {'array': right_depth,  'shape': right_depth.shape, 'type': 'rgb',
			# 	'read_time': received_time, 'serial_number': self._serial_number + '/right'}

		# if self.pointcloud:
		# 	self._cam.retrieve_measure(self._left_pointcloud, sl.MEASURE.XYZRGBA)
		# 	self._cam.retrieve_measure(self._right_pointcloud, sl.MEASURE.XYZRGBA_RIGHT)

		# 	left_pointcloud = self._left_pointcloud.get_data().copy()
		# 	right_pointcloud = self._right_pointcloud.get_data().copy()

		# 	dict_3 = {
		# 		"array": left_pointcloud,
		# 		"shape": left_pointcloud.shape,
		# 		"type": "pointcloud",
		# 		"read_time": received_time,
		# 		"serial_number": self._serial_number + "/left",
		# 	}
		# 	# dict_3 = {'array': right_pointcloud, 'shape': right_pointcloud.shape, 'type': 'pointcloud',
		# 	# 'read_time': received_time, 'serial_number': self._serial_number + '/right'}

		return [dict_1, dict_2]

	def disable_camera(self):
		if hasattr(self, "_cam"):
			self._current_params = None
			self._cam.close()

	# def read_camera(self):
	# 	# Get a new frame from camera
	# 	retval, frame = self._device.read()
	# 	if not retval: return None

	# 	# Extract left and right images from side-by-side
	# 	read_time = time.time()
	# 	left_img, right_img = np.split(frame, 2, axis=1)
	# 	left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
	# 	right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

	# 		# 	self._cam.retrieve_measure(self._left_depth, sl.MEASURE.DEPTH, resolution=self.resolution)
	# 	# 	self._cam.retrieve_measure(self._right_depth, sl.MEASURE.DEPTH_RIGHT, resolution=self.resolution)
	# 	# 	data_dict['depth'] = {
	# 	# 		self._serial_number + '_left': self._left_depth.get_data().copy(),
	# 	# 		self._serial_number + '_right': self._right_depth.get_data().copy()}

	# 	# left_img = cv2.resize(left_img, dsize=(128, 96), interpolation=cv2.INTER_AREA)
	# 	# right_img = cv2.resize(right_img, dsize=(128, 96), interpolation=cv2.INTER_AREA)

	# 	dict_1 = {'array': left_img, 'shape': left_img.shape, 'type': 'rgb',
	# 		'read_time': read_time, 'serial_number': self._serial_number + '/left'}
	# 	dict_2 = {'array': right_img,  'shape': right_img.shape, 'type': 'rgb',
	# 		'read_time': read_time, 'serial_number': self._serial_number + '/right'}

	# 	return [dict_1, dict_2]

	# def disable_camera(self):
	# 	self._device.release()
