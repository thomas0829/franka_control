# https://github.com/ahmeda14960/iris_robots/blob/main/camera_utils/zed_camera.py

import cv2
import numpy as np
import time

def gather_zed_cameras(max_ind=20):
	all_zed_cameras = []
	for i in range(max_ind):
		device = cv2.VideoCapture(i)
		if device.read()[0]:
			correct_w = int(device.get(3)) == 1344
			correct_h = int(device.get(4)) == 376
			if correct_w and correct_h:
				camera = ZedCamera(device)
				all_zed_cameras.append(camera)
			else:
				device.release()
	return all_zed_cameras

class ZedCamera:
	def __init__(self, device):
		# Set the video resolution to HD720 (2560*720)
		device.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
		device.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		self._device = device
		self._serial_number = str(camera.serial_number)

	def read_camera(self):
		# Get a new frame from camera
		retval, frame = self._device.read()
		if not retval: return None

		# Extract left and right images from side-by-side
		read_time = time.time()
		left_img, right_img = np.split(frame, 2, axis=1)
		left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
		right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)



			# 	self._cam.retrieve_measure(self._left_depth, sl.MEASURE.DEPTH, resolution=self.resolution)
        # 	self._cam.retrieve_measure(self._right_depth, sl.MEASURE.DEPTH_RIGHT, resolution=self.resolution)
        # 	data_dict['depth'] = {
        # 		self.serial_number + '_left': self._left_depth.get_data().copy(),
        # 		self.serial_number + '_right': self._right_depth.get_data().copy()}
		
		# left_img = cv2.resize(left_img, dsize=(128, 96), interpolation=cv2.INTER_AREA)
		# right_img = cv2.resize(right_img, dsize=(128, 96), interpolation=cv2.INTER_AREA)

		dict_1 = {'array': left_img, 'shape': left_img.shape, 'type': 'rgb',
			'read_time': read_time, 'serial_number': self._serial_number + '/left'}
		dict_2 = {'array': right_img,  'shape': right_img.shape, 'type': 'rgb',
			'read_time': read_time, 'serial_number': self._serial_number + '/right'}
		
		return [dict_1, dict_2]

	def disable_camera(self):
		self._device.release()