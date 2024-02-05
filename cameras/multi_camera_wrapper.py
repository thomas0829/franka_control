from cameras.camera_thread import CameraThread
from cameras.realsense_camera import gather_realsense_cameras
from cameras.zed_camera import gather_zed_cameras
import time
import cv2

class MultiCameraWrapper:

	def __init__(self, specific_cameras=None, use_threads=True, type="realsense", num_expected_cameras=None):
		"""
		Multi Camera Wrapper that returns images in order of serial number
		params:
		specific_cameras: list of camera objects
		use_threads: bool, whether to use threads for reading the cameras
		type: str, type of camera to use. Options are 'realsense' and 'zed'
		:num_expected_cameras: int, number of cameras expected. If none, won't have default cameras
		"""
		
		self.specific_cameras = specific_cameras
		self.use_threads = use_threads
		self.type = type
		self.num_expected_cameras = num_expected_cameras
		self._reset_cameras()
		self.num_cameras = len(self._all_cameras)
		if self.num_expected_cameras is not None:
			assert self.num_cameras == self.num_expected_cameras, f'Expected {self.num_expected_cameras} cameras, got {self.num_cameras}'


	def _reset_cameras(self):
		"""
		Resets the cameras to the state they were in when the object was first created
		"""
		if hasattr(self, '_all_cameras'):
			for camera in self._all_cameras:
				try:
					camera.disable_camera()
				except RuntimeError as e:
					# This means that the camera is not plugged in right now
					continue
		self._all_cameras = []

		if self.specific_cameras is not None:
			self._all_cameras.extend(self.specific_cameras)
		else:
			if self.type == 'realsense':
				self._all_cameras.extend(gather_realsense_cameras())
			elif self.type == 'zed':
				time.sleep(1)
				self._all_cameras.extend(gather_zed_cameras())
			else: 
				raise NotImplementedError('Type of camera not implemented. You requested no specific cameras with no type')
			
			assert hasattr(self, "num_cameras") or len(self._all_cameras) > 0, 'No cameras found'
		self._sort_cams_by_serial()

		if self.use_threads:
			for i, camera in enumerate(self._all_cameras):
				self._all_cameras[i] = CameraThread(camera)
			time.sleep(1)
	

	def read_cameras(self):
		try:
			all_frames = []
			for camera in self._all_cameras:
				curr_feed = camera.read_camera()
				if curr_feed is not None:
					all_frames.extend(curr_feed)
			if len(all_frames) != self.num_cameras * 2:
				raise RuntimeError(f'Number of cameras changed, expected {self.num_cameras}, got {len(all_frames)//2}')
			return all_frames
		
		except RuntimeError as e:
			print(f'Error reading cameras: {e}')
			input('Press enter when cameras are fixed:\t')
			self._reset_cameras()
			return self.read_cameras()

	def disable_cameras(self):
		for camera in self._all_cameras:
			camera.disable_camera()
	
	def _sort_cams_by_serial(self):
		"""
		Sorts the cameras by serial number
		"""
		self._all_cameras.sort(key=lambda x: x._serial_number)
