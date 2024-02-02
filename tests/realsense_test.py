import time 
import cv2

from cameras.multi_camera_wrapper import MultiCameraWrapper

multi_camera_wrapper = MultiCameraWrapper(type='realsense', use_threads=False)

num_cameras = multi_camera_wrapper.num_cameras

print(f'Number of cameras: {num_cameras}')

while True:
    frames = multi_camera_wrapper.read_cameras()
    for i in range(0, len(frames), 2):
        rgb = frames[i]['array']
        depth = frames[i+1]['array']
        cv2.imshow(f'RGB_cam_{i//2}', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow(f'Depth_cam_{i//2}', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))
        k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        break
    else:
        pass
