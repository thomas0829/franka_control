import colorsys
import math
import os
from parser import Parser

import cv2
import matplotlib.cm as cm
import mujoco
import numpy as np
import nvisii
import open3d as o3d
from base import Renderer

from utils.transformations import quat_to_rmat, rmat_to_quat
from utils.transformations_mujoco import mat_to_quat_mujoco, quat_to_mat_mujoco

EPS = np.finfo(float).eps * 4.0

def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]

def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )
# from robosuite.utils import transform_utils as T
def pose2mat(pose):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose (2-tuple): a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat
# from robosuite.utils.mjcf_utils import xml_path_completion


np.set_printoptions(threshold=np.inf)


class NVISIIRenderer(Renderer):
    def __init__(
        self,
        env,
        img_path="images/",
        width=500,
        height=500,
        spp=256,
        use_noise=False,
        debug_mode=False,
        video_mode=False,
        video_path="videos/",
        video_name="robosuite_video_0.mp4",
        video_fps=60,
        verbose=1,
        vision_modalities=None,
    ):
        """
        Initializes the nvisii wrapper. Wrapping any MuJoCo environment in this
        wrapper will use the NVISII wrapper for rendering.

        Args:
            env (MujocoEnv instance): The environment to wrap.

            img_path (string): Path to images.

            width (int, optional): Width of the rendered image. Defaults to 500.

            height (int, optional): Height of the rendered image. Defaults to 500.

            spp (int, optional): Sample-per-pixel for each image. Larger spp will result
                                 in higher quality images but will take more time to render
                                 each image. Higher quality images typically use an spp of
                                 around 512.

            use_noise (bool, optional): Use noise or denoise. Deafults to false.

            debug_mode (bool, optional): Use debug mode for nvisii. Deafults to false.

            video_mode (bool, optional): By deafult, the NVISII wrapper saves the results as
                                         images. If video_mode is set to true, a video is
                                         produced and will be stored in the directory defined
                                         by video_path. Defaults to false.

            video_path (string, optional): Path to store the video. Required if video_mode is
                                           set to true. Defaults to 'videos/'.

            video_name (string, optional): Name for the file for the video. Defaults to
                                           'robosuite_video_0.mp4'.

            video_fps (int, optional): Frames per second for video. Defaults to 60.

            verbose (int, optional): If verbose is set to 1, the wrapper will print the image
                                     number for each image rendered. If verbose is set to 0,
                                     nothing will be printed. Defaults to 1.

            vision_modalities (string, optional): Options to render image with different ground truths
                                              for NVISII. Options include "normal", "texture_coordinates",
                                              "position", "depth".
        """

        super().__init__(env, renderer_type="nvisii")

        self.env = env
        self.img_path = img_path
        self.width = width
        self.height = height
        self.spp = spp
        self.use_noise = use_noise

        self.video_mode = video_mode
        self.video_path = video_path
        self.video_name = video_name
        self.video_fps = video_fps

        self.verbose = verbose
        self.vision_modalities = vision_modalities

        self.img_cntr = 0

        # env._setup_references()

        # enable interactive mode when debugging
        if debug_mode:
            nvisii.initialize_interactive()
        else:
            nvisii.initialize(headless=True)

        self.segmentation_type = [None] # self.env.camera_segmentations

        # add denoiser to nvisii if not using noise
        if not use_noise:
            nvisii.configure_denoiser()
            nvisii.enable_denoiser()
            nvisii.configure_denoiser(True, True, False)

        if not os.path.exists(img_path):
            os.makedirs(img_path)

        if video_mode:
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            self.video = cv2.VideoWriter(
                video_path + video_name, cv2.VideoWriter_fourcc(*"MP4V"), video_fps, (self.width, self.height)
            )
            print(f"video mode enabled")

        # if vision_modalities is None and self.segmentation_type[0] == None:
        #     nvisii.sample_pixel_area(x_sample_interval=(0.0, 1.0), y_sample_interval=(0.0, 1.0))
        # else:
        #     nvisii.sample_pixel_area(x_sample_interval=(0.5, 0.5), y_sample_interval=(0.5, 0.5))
        nvisii.sample_pixel_area(x_sample_interval=(0.0, 1.0), y_sample_interval=(0.0, 1.0))

        self._init_nvisii_components()

    def _init_nvisii_components(self):
        self._init_lighting()
        self._init_floor(image="light-wood.png")
        self._init_walls(image="gray-plaster.png")
        self._init_camera()

        self._load()

    def _init_lighting(self):
        # Intiailizes the lighting
        self.light_1 = nvisii.entity.create(
            name="light",
            mesh=nvisii.mesh.create_sphere("light"),
            transform=nvisii.transform.create("light"),
        )

        self.light_1.set_light(nvisii.light.create("light"))

        self.light_1.get_light().set_intensity(150)  # intensity of the light
        self.light_1.get_transform().set_scale(nvisii.vec3(0.3))  # scale the light down
        self.light_1.get_transform().set_position(nvisii.vec3(3, 3, 4))  # sets the position of the light
        # self.light_1.get_transform().set_position(nvisii.vec3(3, 3, 4))  # sets the position of the light

    def _init_floor(self, image):
        """
        Intiailizes the floor

        Args:
            image (string): String for the file to use as an image for the floor

        """
        floor_mesh = nvisii.mesh.create_plane(name="plane", size=nvisii.vec2(3, 3))

        floor_entity = nvisii.entity.create(
            name="floor",
            mesh=floor_mesh,
            material=nvisii.material.create("plane"),
            transform=nvisii.transform.create("plane"),
        )
        floor_entity.get_transform().set_scale(nvisii.vec3(1))
        # floor_entity.get_transform().set_position(nvisii.vec3(0, 0, 0))
        # table is at z=0, floor is at z=-0.6
        floor_entity.get_transform().set_position(nvisii.vec3(0, 0, -0.6))

        texture_image = os.path.join("robot/sim/mujoco/assets", "texture", image)
        texture = nvisii.texture.create_from_file(name="floor_texture", path=texture_image)

        floor_entity.get_material().set_base_color_texture(texture)

    def _init_walls(self, image):
        """
        Intializes the walls

        Args:
            image (string): String for the file to use as an image for the walls
        """
        texture_image = os.path.join("robot/sim/mujoco/assets", "texture", image)
        texture = nvisii.texture.create_from_file(name="wall_texture", path=texture_image)

        for geom_name in [mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(self.env.model.ngeom)]:
            if geom_name is not None and "wall" in geom_name:
        
                geom_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                size = self.env.model.geom_size[geom_id]

                pos, quat = self._get_orientation_geom(geom_id)

                wall_entity = nvisii.entity.create(
                    name=geom_name,
                    mesh=nvisii.mesh.create_box(name=geom_name, size=nvisii.vec3(size[0], size[1], size[2])),
                    transform=nvisii.transform.create(geom_name),
                    material=nvisii.material.create(geom_name),
                )

                wall_entity.get_transform().set_position(nvisii.vec3(pos[0], pos[1], pos[2]))

                wall_entity.get_transform().set_rotation(nvisii.quat(quat[0], quat[1], quat[2], quat[3]))

                wall_entity.get_material().set_base_color_texture(texture)

    def _init_camera(self):
        """
        Intializes the camera for the NVISII renderer
        """

        # intializes the camera
        self.camera = nvisii.entity.create(
            name="camera",
            transform=nvisii.transform.create("camera_transform"),
        )

        self.camera.set_camera(
            nvisii.camera.create_from_fov(
                name="camera_camera", field_of_view=1, aspect=float(self.width) / float(self.height)
            )
        )

        # Sets the primary camera of the renderer to the camera entity
        nvisii.set_camera_entity(self.camera)
        self._camera_configuration(
            at_vec=nvisii.vec3(0, 0, 0.06),
            up_vec=nvisii.vec3(0, 0, 1),
            # eye_vec=nvisii.vec3(2.24, 0.0, 2.35),
            eye_vec=nvisii.vec3(1.24, 0.0, 1.35),
            quat=nvisii.quat(-1, 0, 0, 0),
        )

        # Environment configuration
        self._dome_light_intensity = 1
        nvisii.set_dome_light_intensity(self._dome_light_intensity)
        nvisii.set_max_bounce_depth(4)

    def _camera_configuration(self, at_vec, up_vec, eye_vec, quat):
        """
        Sets the configuration for the NVISII camera. Configuration
        is dependent on where the camera is located and where it
        looks at
        """
        # configures the camera
        self.camera.get_transform().look_at(
            at=at_vec, up=up_vec, eye=eye_vec, previous=False  # look at (world coordinate)  # up vector
        )

        self.camera.get_transform().rotate_around(eye_vec, quat)

    def set_camera_pos_quat(self, pos, at):
        self.camera.get_transform().set_position(pos)
        
        # def convert_to_look_at(pos, quat=None, rot_matrix=None):
        #     if quat is not None:
        #         rot_matrix = quat_to_mat_mujoco(quat)
        #     assert rot_matrix is not None, "Either quat or rot_matrix must be provided"
            
        #     eye_vec = pos
        #     forward_vec = np.array([0, 0, 1]) # Adjust based on your coordinate system
        #     up_vec_base = np.array([0, 1, 0])
            
        #     at_vec = eye_vec + np.dot(rot_matrix, forward_vec)
        #     up_vec = np.dot(rot_matrix, up_vec_base)
            
        #     return at_vec, up_vec, eye_vec

        # at_vec, up_vec, eye_vec = convert_to_look_at(pos, rot_matrix=rmat)
        # self.camera.get_transform().look_at(at=at_vec, up=up_vec, eye=eye_vec, previous=False)

        self.camera.get_transform().look_at(
            at=at, up=(0, 0, 1), eye=pos, previous=False  # look at (world coordinate)  # up vector
        )
        # self.camera.get_transform().rotate_around(pos, quat)
        # self.camera.get_transform().set_rotation(quat)

    def _get_orientation_geom(self, geom_id):
        """
        Gets the position and quaternion for a geom
        """

        pos = self.env.data.geom_xpos[geom_id]
        R = self.env.data.geom_xmat[geom_id].reshape(3, 3)

        quat_xyzw = mat2quat(R)
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return pos, quat

    def _load(self):
        """
        Loads the nessecary textures, materials, and geoms into the
        NVISII renderer
        """
        self.parser = Parser("nvisii", self.env, self.segmentation_type)
        self.parser.parse_textures()
        self.parser.parse_materials()
        self.parser.parse_geometries()
        self.components = self.parser.components
        # self.max_elements = self.parser.max_elements
        # self.max_instances = self.parser.max_instances
        # self.max_classes = self.parser.max_classes

    def update(self):
        """
        Updates the states for the wrapper given a certain action

        Args:
            action (np-array): The action the robot should take
        """
        for key, value in self.components.items():
            self._update_orientation(name=key, component=value)

    def _update_orientation(self, name, component):
        """
        Update position for an object or a robot in renderer.

        Args:
            name (string): name of component
            component (nvisii entity or scene): Object in renderer and other info
                                                for object.
        """

        obj = component.obj
        parent_body_name = component.parent_body_name
        geom_pos = component.geom_pos
        geom_quat = component.geom_quat
        dynamic = component.dynamic

        if not dynamic:
            return

        self.body_tags = ["link", "finger", "hand", "table"]#, "table"]# , "robot", "pedestal", "gripper", "peg"]
        
        # if parent_body_name != "worldbody":
        body_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, parent_body_name
            )
        if self.tag_in_name(name):
            pos = self.env.data.xpos[body_id]
        else:
            geom_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            pos = self.env.data.geom_xpos[geom_id]

        # if parent_body_name != "worldbody":
        B = self.env.data.xmat[body_id].reshape((3, 3))
        quat_xyzw_body = mat2quat(B)
        quat_wxyz_body = np.array(
            [quat_xyzw_body[3], quat_xyzw_body[0], quat_xyzw_body[1], quat_xyzw_body[2]]
        )  # wxyz
        nvisii_quat = nvisii.quat(*quat_wxyz_body) * nvisii.quat(*geom_quat)

        if self.tag_in_name(name):
            # Add position offset if there are position offset defined in the geom tag
            homo_mat = pose2mat((np.zeros((1, 3), dtype=np.float32), quat_xyzw_body))
            pos_offset = homo_mat @ np.array([geom_pos[0], geom_pos[1], geom_pos[2], 1.0]).transpose()
            pos = pos + pos_offset[:3]

        # else:
        #     pos = [0, 0, 0]
        #     nvisii_quat = nvisii.quat(1, 0, 0, 0)

        # if parent_body_name != "worldbody":

            

        #     # if self.tag_in_name(name):
        #     #     body_id = mujoco.mj_name2id(
        #     #         self.env.model, mujoco.mjtObj.mjOBJ_BODY, name
        #     #     )
        #     #     pos = self.env.data.xpos[body_id]
        #     # else:
        #     geom_id = mujoco.mj_name2id(
        #         self.env.model, mujoco.mjtObj.mjOBJ_GEOM, name
        #     )
        #     pos = self.env.data.geom_xpos[geom_id]


        #     body_id = mujoco.mj_name2id(
        #         self.env.model, mujoco.mjtObj.mjOBJ_BODY, name
        #     )
        #     B = self.env.data.xmat[body_id].reshape((3, 3))
            
        #     B = self.env.data.geom_xmat[geom_id].reshape((3, 3))
        #     # quat_wxyz_body = mat_to_quat_mujoco(B)
        #     quat_xyzw_body = rmat_to_quat(B)
        #     quat_wxyz_body = np.array(
        #         [quat_xyzw_body[3], quat_xyzw_body[0], quat_xyzw_body[1], quat_xyzw_body[2]]
        #     )  # wxyz
        #     nvisii_quat = nvisii.quat(*quat_wxyz_body) * nvisii.quat(*geom_quat)

        #     # if self.tag_in_name(name):
        #     # Add position offset if there are position offset defined in the geom tag
        #     homo_mat = pose2mat((np.zeros((1, 3), dtype=np.float32), quat_xyzw_body))
        #     pos_offset = homo_mat @ np.array([geom_pos[0], geom_pos[1], geom_pos[2], 1.0]).transpose()
        #     pos = pos + pos_offset[:3]

        # else:
        #     pos = [-1, 0, 0]
        #     nvisii_quat = nvisii.quat(1, 0, 0, 0)  # wxyz

        if isinstance(obj, nvisii.scene):
            
            # temp fix -- look into XML file for correct quat
            if "s_visual" in name:
                # single robot
                if len(self.env.robots) == 1:
                    nvisii_quat = nvisii.quat(0, 0.5, 0, 0)
                # two robots - 0
                elif len(self.env.robots) == 2 and "robot_0" in name:
                    nvisii_quat = nvisii.quat(-0, 0.5, 0.5, 0)
                # two robots - 1
                else:
                    nvisii_quat = nvisii.quat(-0, 0.5, -0.5, 0)

            obj.transforms[0].set_position(nvisii.vec3(pos[0], pos[1], pos[2]))
            obj.transforms[0].set_rotation(nvisii_quat)
        else:
            obj.get_transform().set_position(nvisii.vec3(pos[0], pos[1], pos[2]))
            obj.get_transform().set_rotation(nvisii_quat)

    def tag_in_name(self, name):
        """
        Checks if one of the tags in body tags in the name

        Args:
            name (string): Name of component
        """
        for tag in self.body_tags:
            if tag in name:
                return True
        return False

    def render(self, render_type="png"):
        """
        Renders an image of the NVISII renderer

        Args:
            render_type (string, optional): Type of file to save as. Defaults to 'png'
        """

        self.img_cntr += 1
        verbose_word = "frame" if self.video_mode else "image"

        if self.video_mode:
            img_file = f"{self.img_path}/image_0.{render_type}"
            if self.segmentation_type[0] != None:
                self.render_segmentation_data(img_file)
            elif self.vision_modalities is None:
                self.render_to_file(img_file)
            else:
                self.render_data_to_file(img_file)

            self.video.write(cv2.imread(img_file))
        else:
            img_file = f"{self.img_path}/image_{self.img_cntr}.{render_type}"
            if self.segmentation_type[0] != None:
                self.render_segmentation_data(img_file)
            elif self.vision_modalities is None:
                self.render_to_file(img_file)
            else:
                self.render_data_to_file(img_file)

        if self.verbose == 1:
            print(f"Rendering {verbose_word}... {self.img_cntr}")

    def render_to_file(self, img_file):
        nvisii.render_to_file(width=self.width, height=self.height, samples_per_pixel=self.spp, file_path=img_file)

    def render_segmentation_data(self, img_file):

        segmentation_array = nvisii.render_data(
            width=int(self.width),
            height=int(self.height),
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="entity_id",
            seed=1,
        )
        segmentation_array = np.array(segmentation_array).reshape(self.height, self.width, 4)[:, :, 0]
        segmentation_array[segmentation_array > 3.4028234663852886e37] = 0
        segmentation_array[segmentation_array < 3.4028234663852886e-37] = 0
        segmentation_array = np.flipud(segmentation_array)

        rgb_data = self.segmentation_to_rgb(segmentation_array.astype(dtype=np.uint8))

        from PIL import Image

        rgb_img = Image.fromarray(rgb_data)
        rgb_img.save(img_file)

    def render_data_to_file(self, img_file):

        if self.vision_modalities == "depth" and self.img_cntr != 1:

            depth_data = nvisii.render_data(
                width=self.width,
                height=self.height,
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options=self.vision_modalities,
            )

            depth_data = np.array(depth_data).reshape(self.height, self.width, 4)
            depth_data = np.flipud(depth_data)[:, :, [0, 1, 2]]

            # normalize depths
            depth_data[:, :, 0] = (depth_data[:, :, 0] - np.min(depth_data[:, :, 0])) / (
                np.max(depth_data[:, :, 0]) - np.min(depth_data[:, :, 0])
            )
            depth_data[:, :, 1] = (depth_data[:, :, 1] - np.min(depth_data[:, :, 1])) / (
                np.max(depth_data[:, :, 1]) - np.min(depth_data[:, :, 1])
            )
            depth_data[:, :, 2] = (depth_data[:, :, 2] - np.min(depth_data[:, :, 2])) / (
                np.max(depth_data[:, :, 2]) - np.min(depth_data[:, :, 2])
            )

            from PIL import Image

            depth_image = Image.fromarray(((1 - depth_data) * 255).astype(np.uint8))
            depth_image.save(img_file)

        elif self.vision_modalities == "normal" and self.img_cntr != 1:

            normal_data = nvisii.render_data(
                width=self.width,
                height=self.height,
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="screen_space_normal",
            )

            normal_data = np.array(normal_data).reshape(self.height, self.width, 4)
            normal_data = np.flipud(normal_data)[:, :, [0, 1, 2]]

            normal_data[:, :, 0] = (normal_data[:, :, 0] + 1) / 2 * 255  # R
            normal_data[:, :, 1] = (normal_data[:, :, 1] + 1) / 2 * 255  # G
            normal_data[:, :, 2] = 255 - ((normal_data[:, :, 2] + 1) / 2 * 255)  # B

            from PIL import Image

            normal_image = Image.fromarray((normal_data).astype(np.uint8))
            normal_image.save(img_file)

        else:

            nvisii.render_data_to_file(
                width=self.width,
                height=self.height,
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options=self.vision_modalities,
                file_path=img_file,
            )

    def randomize_colors(self, N, bright=True):
        """
        Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py#L59
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.5
        hsv = [(1.0 * i / N, 1, brightness) for i in range(N)]
        colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
        rstate = np.random.RandomState(seed=20)
        np.random.shuffle(colors)
        return colors

    def segmentation_to_rgb(self, seg_im, random_colors=False):
        """
        Helper function to visualize segmentations as RGB frames.
        NOTE: assumes that geom IDs go up to 255 at most - if not,
        multiple geoms might be assigned to the same color.
        """
        # ensure all values lie within [0, 255]
        seg_im = np.mod(seg_im, 256)

        if random_colors:
            colors = self.randomize_colors(N=256, bright=True)
            return (255.0 * colors[seg_im]).astype(np.uint8)
        else:

            cmap = cm.get_cmap("jet")

            max_r = 0
            if self.segmentation_type[0][0] == "element":
                max_r = np.amax(seg_im) + 1
            elif self.segmentation_type[0][0] == "class":
                max_r = self.max_classes
                for i in range(len(seg_im)):
                    for j in range(len(seg_im[0])):
                        if seg_im[i][j] in self.parser.entity_id_class_mapping:
                            seg_im[i][j] = self.parser.entity_id_class_mapping[seg_im[i][j]]
                        else:
                            seg_im[i][j] = max_r - 1
            elif self.segmentation_type[0][0] == "instance":
                max_r = self.max_instances
                for i in range(len(seg_im)):
                    for j in range(len(seg_im[0])):
                        if seg_im[i][j] in self.parser.entity_id_class_mapping:
                            seg_im[i][j] = self.parser.entity_id_class_mapping[seg_im[i][j]]
                        else:
                            seg_im[i][j] = max_r - 1

            color_list = np.array([cmap(i / (max_r)) for i in range(max_r)])

            return (color_list[seg_im] * 255).astype(np.uint8)

    def reset(self):
        nvisii.clear_all()
        self._init_nvisii_components()
        self.update()

    def get_pixel_obs(self):
        frame_buffer = nvisii.render(width=self.width, height=self.height, samples_per_pixel=self.spp)

        frame_buffer = np.array(frame_buffer).reshape(self.height, self.width, 4)
        frame_buffer = np.flipud(frame_buffer)

        return frame_buffer

    def close(self):
        """
        Deinitializes the nvisii rendering environment
        """
        nvisii.deinitialize()