import os
import xml.etree.ElementTree as ET
from collections import namedtuple

import numpy as np
import nvisii
from base_parser import BaseParser
from nvisii_utils import load_object


def string_to_array(string):
    """
    Converts a array string in mujoco xml to np.array.

    Examples:
        "0 1 2" => [0, 1, 2]

    Args:
        string (str): String to convert to an array

    Returns:
        np.array: Numerical array equivalent of @string
    """
    return np.array([float(x) for x in string.strip().split(" ")])


Components = namedtuple(
    "Components",
    [
        "obj",
        "geom_index",
        "element_id",
        "parent_body_name",
        "geom_pos",
        "geom_quat",
        "dynamic",
    ],
)


class Parser(BaseParser):
    def __init__(self, renderer, env, segmentation_type):
        """
        Parse the mujoco xml and initialize NVISII renderer objects.
        Args:
            env (Mujoco env): Environment to parse
        """

        super().__init__(renderer, env)
        self.segmentation_type = segmentation_type
        # self.create_class_mapping()
        self.components = {}

    def parse_textures(self):
        """
        Parse and load all textures and store them
        """

        self.texture_attributes = {}
        self.texture_id_mapping = {}

        for texture in self.xml_root.iter("texture"):
            texture_type = texture.get("type")
            texture_name = texture.get("name")
            texture_file = texture.get("file")
            texture_rgb = texture.get("rgb1")

            if texture_file is not None:
                self.texture_attributes[texture_name] = texture.attrib
            else:
                color = np.array(string_to_array(texture_rgb))
                self.texture_id_mapping[texture_name] = (color, texture_type)

    def parse_materials(self):
        """
        Parse all materials and use texture mapping to initialize materials
        """

        self.material_texture_mapping = {}
        for material in self.xml_root.iter("material"):
            material_name = material.get("name")
            texture_name = material.get("texture")
            self.material_texture_mapping[material_name] = texture_name

    def parse_meshes(self):
        """
        Create mapping of meshes.
        """
        self.meshes = {}
        for mesh in self.xml_root.iter("mesh"):
            self.meshes[mesh.get("name")] = mesh.attrib

    def parse_geometries(self):
        """
        Iterate through each goemetry and load it in the NVISII renderer.
        """
        self.parse_meshes()
        element_id = 0
        repeated_names = {}

        self.entity_id_class_mapping = {}

        # ###
        # geom_name="sphere"
        # geom_pos = [0.1, 0.1, 0.1]
        # geom_quat = [1, 0, 0, 0]
        # obj, entity_ids = load_object(
        #         geom=None,
        #         geom_name=geom_name,
        #         geom_type="sphere",
        #         geom_quat=geom_quat,
        #         geom_pos=geom_pos,
        #         geom_size=[.5, .1, .1],
        #         geom_scale=None,
        #         geom_rgba=[1, 0, 0, 1],
        #         geom_tex_name=None,
        #         geom_tex_file=None,
        #         meshes=self.meshes,
        #     )

        # geom_index = 0
        # parent_body_name = "worldbody"
        # dynamic = False
        # self.components[geom_name] = Components(
        #         obj=obj,
        #         geom_index=geom_index,
        #         element_id=element_id,
        #         parent_body_name=parent_body_name,
        #         geom_pos=geom_pos,
        #         geom_quat=geom_quat,
        #         dynamic=dynamic,
        #     )
        # return
        # ###

        materials = {}
        for idx, mat in enumerate(self.xml_root.iter("material")):
            materials[mat.get("name")] = mat

        for geom_index, geom in enumerate(self.xml_root.iter("geom")):

            parent_body = self.parent_map.get(geom)
            parent_body_name = parent_body.get("name", "worldbody")

            geom_name = geom.get("name")
            if geom_name is None:# or not ("link0" in geom_name or "link1" in geom_name):
                continue
            geom_type = geom.get("type", "sphere")

            # if geom_type == "mesh" and geom.get("class") == "visual":
            #     print(geom.get("name"))
            # print("class", geom.get("class"))
            rgba_str = geom.get("rgba")
            geom_rgba = string_to_array(rgba_str) if rgba_str is not None else None

            if geom_name is None:
                if parent_body_name in repeated_names:
                    geom_name = parent_body_name + str(repeated_names[parent_body_name])
                    repeated_names[parent_body_name] += 1
                else:
                    geom_name = parent_body_name + "0"
                    repeated_names[parent_body_name] = 1

            if "mesh" in geom.attrib.keys():
                geom_type = "mesh"
            # if (geom.get("group") != "1" and geom_type != "plane") or ("collision" in geom_name):
            # remove collision meshes
            # if "collision" in geom_name:
            if geom.get("class") == "collision":
                continue
            if geom_type == "mesh" and "mesh" not in geom.attrib.keys():
                continue
            if "floor" in geom_name or "wall" in geom_name:
                continue
            # if geom_type != "mesh":
            #     continue

            geom_quat = string_to_array(geom.get("quat", "1 0 0 0"))
            geom_quat = [geom_quat[0], geom_quat[1], geom_quat[2], geom_quat[3]]

            geom_pos = string_to_array(geom.get("pos", "0 0 0"))

            if "mesh" in geom.attrib.keys() and geom.get("mesh") is not None:
                geom_scale = string_to_array(
                    self.meshes[geom.get("mesh")].get("scale", "1 1 1")
                )
            else:
                geom_scale = [1, 1, 1]
                
            geom_size = string_to_array(geom.get("size", "1 1 1"))

            geom_mat = geom.get("material")

            dynamic = True

            geom_tex_name = None
            geom_tex_file = None

            if geom_mat is not None:
                
                geom_tex_name = self.material_texture_mapping[geom_mat]
                # has texture file
                if geom_tex_name in self.texture_attributes:
                    geom_tex_file = self.texture_attributes[geom_tex_name]["file"]
                # has texture color
                else:
                    geom_rgba = materials[geom_mat].get("rgba")
                    if geom_rgba is not None:
                        geom_rgba = np.array(geom_rgba.split(" ")).astype(np.float)
            class_id = self.get_class_id(geom_index, element_id)

            # if geom_name != "vention_table":
            #     continue
            # load obj into nvisii
            obj, entity_ids = load_object(
                geom=geom,
                geom_name=geom_name,
                geom_type=geom_type,
                geom_quat=geom_quat,
                geom_pos=geom_pos,
                geom_size=geom_size,
                geom_scale=geom_scale,
                geom_rgba=geom_rgba,
                geom_tex_name=geom_tex_name,
                geom_tex_file=geom_tex_file,
                meshes=self.meshes,
            )

            element_id += 1

            for entity_id in entity_ids:
                self.entity_id_class_mapping[entity_id] = class_id

            self.components[geom_name] = Components(
                obj=obj,
                geom_index=geom_index,
                element_id=element_id,
                parent_body_name=parent_body_name,
                geom_pos=geom_pos,
                geom_quat=geom_quat,
                dynamic=dynamic,
            )

        self.max_elements = element_id

    def create_class_mapping(self):
        """
        Create class name to index mapping for both semantic and instance
        segmentation.
        """
        self.class2index = {}
        for i, c in enumerate(self.env.model._classes_to_ids.keys()):
            self.class2index[c] = i
        self.class2index[None] = i + 1
        self.max_classes = len(self.class2index)

        self.instance2index = {}
        for i, instance_class in enumerate(self.env.model._instances_to_ids.keys()):
            self.instance2index[instance_class] = i
        self.instance2index[None] = i + 1
        self.max_instances = len(self.instance2index)

    def get_class_id(self, geom_index, element_id):
        """
        Given index of the geom object get the class id based on
        self.segmentation type.
        """

        if (
            self.segmentation_type[0] == None
            or self.segmentation_type[0][0] == "element"
        ):
            class_id = element_id
        elif self.segmentation_type[0][0] == "class":
            class_id = self.class2index[
                self.env.model._geom_ids_to_classes.get(geom_index)
            ]
        elif self.segmentation_type[0][0] == "instance":
            class_id = self.instance2index[
                self.env.model._geom_ids_to_instances.get(geom_index)
            ]

        return class_id

    def tag_in_name(self, name, tags):
        """
        Checks if one of the tags in body tags in the name

        Args:
            name (str): Name of geom element.

            tags (array): List of keywords to check from.
        """
        for tag in tags:
            if tag in name:
                return True
        return False
