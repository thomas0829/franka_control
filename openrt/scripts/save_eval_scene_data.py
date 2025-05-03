import os
import time
import pickle

import hydra
import matplotlib.pyplot as plt
import numpy as np
import random

from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

from PIL import Image
import requests
import json_numpy
json_numpy.patch()


from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import logging
import open3d as o3d
from utils.pointclouds import *
from PIL import Image, ImageDraw

# visualize grasp
from utils.pointclouds import depth_to_points, crop_points, visualize_pcds, points_to_pcd
import cv2

# LOGGING
# Setup logger
import logging
from datetime import date

# Setup logger
logger = logging.getLogger("Evaluation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

class LogColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log_config(msg):
    logger.info(f"{LogColors.CYAN}{msg}{LogColors.END}")

def log_connect(msg):
    logger.info(f"{LogColors.BLUE}{msg}{LogColors.END}")

def log_instruction(msg):
    logger.info(f"{LogColors.YELLOW}{msg}{LogColors.END}")

def log_success(msg):
    logger.info(f"{LogColors.GREEN}{msg}{LogColors.END}")

def log_failure(msg):
    logger.info(f"{LogColors.RED}{msg}{LogColors.END}")

def log_important(msg):
    logger.info(f"{LogColors.BOLD}{LogColors.HEADER}{msg}{LogColors.END}")
    

def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

# DEBUG
x = 0
y = 0
# Click event handler
def onclick(event):
    """
    Click event handler for the image
    """
    global x, y
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked coordinates: x={x}, y={y}")


# Load camera info
def load_cam_info(cam_info_file):
    with open(cam_info_file, "rb") as f:
        cam_info = pickle.load(f)
    cam_pose = cam_info[0]["extrinsic_in_world"]
    cam_K = cam_info[0]["intrinsic"]
    return cam_pose, cam_K

# Execute interpolated motion   
def execute_interpolated_motion(env, curr_pose_matrix, target_pose_matrix, obs, distance_threshold=0.01, verbose=False, visualize=False, sleep_time=0.5, close_gripper_state=False):
    """Interpolates poses and executes them sequentially in the environment."""

    # execute pregrasp
    interpolated_motions = interpolate_6d_poses(curr_pose_matrix, target_pose_matrix, distance_threshold=distance_threshold)
    
    
    # prev_gripper = obs["lowdim_ee"][-1] # WARNING: keep the gripper state same
    # if close_gripper:
        

    
    for motion_6d in interpolated_motions[1:]:
        sub_pose = np.zeros(7)
        sub_pose[:6] = motion_6d
        
        if close_gripper_state:
            sub_pose[6] = 1
        else:
            sub_pose[6] = 0 # WARNING: keep the gripper state same


        next_obs, _, _, _ = env.step_with_pose(sub_pose)
        prev_gripper = next_obs["lowdim_ee"][-1]
        
        pose_error = np.linalg.norm(next_obs["lowdim_ee"][:6] - sub_pose[:6])
        position_error = np.linalg.norm(next_obs["lowdim_ee"][:3] - sub_pose[:3])
        
        if verbose:
            logging.info(f"pose error: {pose_error}")
            logging.info(f"position error: {position_error}")
            logging.info(f"------------------------------------------------------------------------------------")
    
    time.sleep(sleep_time)
    return next_obs, interpolated_motions

# Visualize grasp pose
def visualize_grasp_pose(target_pose_matrix, pregraps_pose_matrix, curr_pose_matrix, grasp_predict, image, depth, cam_K, cam_pose, distance_threshold=0.01):
    # visualize target grasp and pregrasp
    target_grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # you can adjust the size if needed
    target_grasp_frame.transform(target_pose_matrix)
    
    pregrasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # you can adjust the size if needed
    pregrasp_frame.transform(pregraps_pose_matrix)
    
    motions_frames = []
    interpolated_motions = interpolate_6d_poses(curr_pose_matrix, pregraps_pose_matrix, distance_threshold=distance_threshold)
    for motion in interpolated_motions:
        motion_matrix = get_pose_matrix(motion)
        motion_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # you can adjust the size if needed
        motion_frame.transform(motion_matrix)
        motions_frames.append(motion_frame)
    
    
    # visualize grasp pose
    from m2t2.molmo_remote_pred import DRAW_POINTS
    draw_points_hom = np.hstack([DRAW_POINTS, np.ones((4, 1))])  # (4,4), add 1 for homogeneous coordinates
    draw_points_world = (rotate_pose_z_axis_counterclockwise(grasp_predict, OFFSET_ANGLE+45) @ draw_points_hom.T).T[:, :3]  # Transform and drop the homogeneous coordinate

    spheres = []
    for p in draw_points_world:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # small sphere
        sphere.paint_uniform_color([1, 0, 0])  # red color
        sphere.translate(p)  # move sphere to the point
        spheres.append(sphere)

    pcds = []
    colors = image.reshape(-1, 3) / 255.0
    points = depth_to_points(depth, cam_K, cam_pose, depth_scale=1000.0)
    points, colors = crop_points(points, colors=colors)
    pcds.append(points_to_pcd(points, colors=colors))
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    # visualize_pcds(pcds + [mesh_frame, target_grasp_frame, pregrasp_frame] + motions_frames + spheres)

    geometries = pcds + [mesh_frame, target_grasp_frame, pregrasp_frame] + motions_frames + spheres
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Grasp Pose Visualization")

    for geom in geometries:
        vis.add_geometry(geom)

    # Register callback: press 'q' to exit
    def close_window(vis):
        vis.close()
        return False  # return False to stop the event loop immediately

    vis.register_key_callback(ord("Q"), close_window)
    vis.register_key_callback(ord("q"), close_window)

    vis.run()
    vis.destroy_window()


from PIL import Image, ImageDraw, ImageFont
def concatenate_with_labels(image1: Image.Image, label1: str,
                             image2: Image.Image, label2: str,
                             image3: Image.Image, label3: str) -> Image.Image:
    """Horizontally concatenate three PIL images with text labels above each. If any image is None, use a black placeholder."""
    
    # Set font
    try:
        font = ImageFont.truetype("arial.ttf", size=20)  # Try using Arial
    except IOError:
        font = ImageFont.load_default()  # Fallback if Arial not found

    # Determine the reference height and width
    ref_image = next(img for img in [image1, image2, image3] if img is not None)
    ref_height = ref_image.height
    ref_width = ref_image.width

    # Replace None images with black placeholders
    def ensure_image(img):
        if img is None:
            return Image.new('RGB', (ref_width, ref_height), color=(0, 0, 0))
        return img

    image1 = ensure_image(image1)
    image2 = ensure_image(image2)
    image3 = ensure_image(image3)

    # Ensure images have the same height
    assert image1.height == image2.height == image3.height, "All images must have the same height"

    # Create dummy image for text sizing
    dummy_img = Image.new('RGB', (10, 10))
    draw = ImageDraw.Draw(dummy_img)

    label1_bbox = draw.textbbox((0, 0), label1, font=font)
    label1_width = label1_bbox[2] - label1_bbox[0]
    label1_height = label1_bbox[3] - label1_bbox[1]

    label2_bbox = draw.textbbox((0, 0), label2, font=font)
    label2_width = label2_bbox[2] - label2_bbox[0]
    label2_height = label2_bbox[3] - label2_bbox[1]

    label3_bbox = draw.textbbox((0, 0), label3, font=font)
    label3_width = label3_bbox[2] - label3_bbox[0]
    label3_height = label3_bbox[3] - label3_bbox[1]

    # Determine text height (max of three labels) + padding
    text_height = max(label1_height, label2_height, label3_height) + 10

    # Create new blank image with space for text and all images
    total_width = image1.width + image2.width + image3.width
    total_height = image1.height + text_height
    new_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))  # white background

    # Paste images
    new_image.paste(image1, (0, text_height))
    new_image.paste(image2, (image1.width, text_height))
    new_image.paste(image3, (image1.width + image2.width, text_height))

    # Draw labels
    draw = ImageDraw.Draw(new_image)

    draw.text(
        (image1.width // 2 - label1_width // 2, (text_height - label1_height) // 2),
        label1, font=font, fill=(0, 0, 0)
    )
    draw.text(
        (image1.width + image2.width // 2 - label2_width // 2, (text_height - label2_height) // 2),
        label2, font=font, fill=(0, 0, 0)
    )
    draw.text(
        (image1.width + image2.width + image3.width // 2 - label3_width // 2, (text_height - label3_height) // 2),
        label3, font=font, fill=(0, 0, 0)
    )

    return new_image


# graspGPT configs
from io import BytesIO
from base64 import b64encode
class GraspGPT:
    def __init__(self, url):
        self.url = url
        self.image_buffer = None
        self.grasp_idx = None

    def reset(self):
        self.image_buffer = None
        self.grasp_idx = None

    def encode_image(self, image):
        self.image_buffer = image
        
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_str = b64encode(image_bytes.getvalue()).decode("utf-8")
        return image_str

    def pred_grasp(self, image_str, depth, cam_K, grasps, task):
        payload = {"image_str": image_str,"depth": depth.tolist(),"cam_K": cam_K.tolist(),"grasps": grasps.tolist(),"task": task}
        response = requests.post(self.url, json=payload)
        data = response.json()
        # print(data)
            
        idx = data["grasp_idx"]
        DRAW_POINTS = np.array([
            [0.041, 0, 0.112],
            [0.041, 0, 0.066],
            [-0.041, 0, 0.066],
            [-0.041, 0, 0.112],
        ])  # in grasp frame

        grasp = grasps[idx]
        draw = ImageDraw.Draw(self.image_buffer)
        draw_points = DRAW_POINTS @ grasp[:3, :3].T + grasp[:3, 3]
        draw_points_px = draw_points @ cam_K.T
        draw_points_px = draw_points_px[:, :2] / draw_points_px[:, 2:3]
        draw_points_px = draw_points_px.round().astype(int).tolist()

        for i in range(len(DRAW_POINTS)-1):
            p0 = draw_points_px[i]
            p1 = draw_points_px[i+1]
            draw.line(p0 + p1, fill="red", width=2)
        
        self.grasp_idx = idx
        return idx
    
    def get_image(self):
        if not hasattr(self, 'image_buffer'):
            self.image_buffer = None
        return self.image_buffer
    
    def get_grasp_idx(self):
        return self.grasp_idx

def release_gripper(env, lowdim_ee, time_sleep=3):
    for _ in range(5):
        sub_pose = np.zeros(7)
        sub_pose[:6] = lowdim_ee[:6]
        sub_pose[6] = 0
        env.step_with_pose(sub_pose)
    time.sleep(time_sleep)

def close_gripper(env, lowdim_ee, time_sleep=3):
    for _ in range(5):
        sub_pose = np.zeros(7)
        sub_pose[:6] = lowdim_ee[:6]
        sub_pose[6] = 1
        env.step_with_pose(sub_pose)
    time.sleep(time_sleep)
    
GRIPPER_Z_OFFSET = 0.177
OFFSET_ANGLE = 135

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

@hydra.main(
    config_path="../../configs/", config_name="eval_graspmo_real", version_base="1.1"
)
def run_experiment(cfg):
    # initialize env
    set_random_seed(cfg.seed)
    assert cfg.robot.imgs, "ERROR: set robot.imgs=true to record image observations!"
    cfg.robot.control_hz = 15
    log_config(f"blocking_control: {cfg.robot.blocking_control}, control_hz: {cfg.robot.control_hz}")
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env) if "env" in cfg.keys() else None,
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )
    camera_names = [k for k in env.get_images().keys()]
    log_config(f"Camera names: {camera_names}")
    env.seed(cfg.seed)
    obs = env.reset()
    home_pose = obs["lowdim_ee"]
    log_success(f"done with reset environment")
    
    
    # load cam config
    cam_pose, cam_K = load_cam_info(cfg.cam_info_file)
    cam_pose[1][3] -= 0.05 # fintune value
    
    
    # load m2t2 model
    from hydra import compose, initialize_config_dir
    from m2t2.m2t2_model import M2T2Model
    from hydra.core.global_hydra import GlobalHydra
    # with initialize_config_dir(config_dir=cfg.m2t2_config_path, version_base='1.3'):
    #     m2t2_cfg = compose(config_name=cfg.m2t2_config_name)
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=cfg.m2t2_config_path, version_base='1.3')

    m2t2_cfg = compose(
        config_name=cfg.m2t2_config_name,
        overrides=[], 
    )
    m2t2_cfg.eval.mask_thresh = cfg.m2t2_mask_thresh
    m2t2_cfg.eval.num_runs = cfg.m2t2_num_runs
    m2t2_cfg.eval.checkpoint = cfg.m2t2_checkpoint
    m2t2_model = M2T2Model(cfg=m2t2_cfg, cam_intrinsics=cam_K)
    

    save_file_name = input("Enter the save file name: ")
    save_data_info_path = f"/home/prior/M2T2/evaluation_info/{save_file_name}"
    
    # mode = input("choose model[graspmo, graspGPT]: ")
    mode = "inference"
    log_config(f"Mode: {mode}")
    visualize = True
    verbose = False
    if mode == "inference":
        # load grasp molmo model
        from m2t2.molmo_remote_pred import GraspMolmoBeaker
        graspMolmo = GraspMolmoBeaker(cfg.grasp_molmo_url)
        
        # load graspGPT model
        graspGPT = GraspGPT(cfg.grasp_gpt_url)
        
        # load vanila molmo model
        # from m2t2.molmo_remote_pred import ZeroShotMolmoModal
        vanilaMolmo = GraspMolmoBeaker(cfg.vanila_molmo_url)
        while True:
            # reset model
            graspMolmo.reset()
            graspGPT.reset()
            vanilaMolmo.reset()
                        
            continue_code = False
            while not continue_code:
                # ask if want to refresh the observation 
                refresh = input("refresh the observation? (y/n): ")
                while refresh not in ["y", "n"]:
                    refresh = input("refresh the observation? (y/n): ")
                if refresh == "y":
                    sub_continue_code = False
                else:
                    sub_continue_code = True
                
                # get image and grasp proposal
                while not sub_continue_code:
                    image = env.get_observation()[f"{camera_names[0]}_rgb"]
                    depth = np.squeeze(env.get_observation()[f"{camera_names[0]}_depth"])
                    # Image.fromarray(image).show()  
                    cv2.imshow("Image Viewer", pil_to_cv2(Image.fromarray(image)))
                    
                    # exit windows
                    while True:
                        key = cv2.waitKey(0)
                        if key == ord('q'):
                            break
                    cv2.destroyAllWindows() 
    
                          
                    # generate grasps proposal (M2T2)
                    log_instruction(f"Generating grasps proposal...")
                    grasps_world, _ = m2t2_model.generate_grasps(cam_pose, depth, image, visualize=visualize, verbose=False)
                    grasps = np.linalg.inv(cam_pose)[None] @ grasps_world
                    
                    sub_continue_code = input("continue? (y/n): ")
                    while sub_continue_code not in ["y", "n"]:
                        sub_continue_code = input("continue? (y/n): ")
                    if sub_continue_code == "y":
                        sub_continue_code = True
                    else:
                        sub_continue_code = False
                
                # save image, depth, pointclouds, grasps, cam_pose, cam_K
                data_info = {
                    "image": image,
                    "depth": depth,
                    "grasps_world": grasps_world,
                    "grasps": grasps,
                    "cam_pose": cam_pose,
                    "cam_K": cam_K,
                }
                with open(save_data_info_path, "wb") as f:
                    pickle.dump(data_info, f)
                log_success(f"Saved data info to {save_data_info_path}")
                break


if __name__ == "__main__":
    run_experiment()




