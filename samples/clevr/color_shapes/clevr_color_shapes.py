"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import json
import os
import re
import sys
import math
import random
from os import listdir
from pathlib import Path

import numpy as np
import cv2
import torch
from cv2 import imread
from torchvision.io import read_image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class CLEVRColorShapeConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "color_shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 9  # background + 3 shapes * 3 colors

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class CLEVRColorShapeDataset(utils.Dataset):
    """Generates the color and shape synthetic dataset. The dataset consists of simple
    colors (red, blue, green) and shapes (cube, sphere, cylinder) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    # Partially based on:
    # https://towardsdatascience.com/object-detection-using-mask-r-cnn-on-a-custom-dataset-4f79ab692f6d
    def load_dataset(self, dataset_dir, is_train=True):

        # Add classes
        self.add_class("color_shapes", 1, "red_cube")
        self.add_class("color_shapes", 2, "red_sphere")
        self.add_class("color_shapes", 3, "red_cylinder")

        self.add_class("color_shapes", 4, "blue_cube")
        self.add_class("color_shapes", 5, "blue_sphere")
        self.add_class("color_shapes", 6, "blue_cylinder")

        self.add_class("color_shapes", 7, "green_cube")
        self.add_class("color_shapes", 8, "green_sphere")
        self.add_class("color_shapes", 9, "green_cylinder")

        # For each image in the dataset
        for filename in listdir(dataset_dir):

            # Skip mask files
            if "mask" in filename:
                continue

            # Name of image file
            name = filename[:-4]

            # Image ID
            image_id = name[10:]

            # Image path
            img_path = dataset_dir + filename

            # Skip all images after 100 if we are building the train set
            if is_train and int(image_id) >= 99:
                continue

            # Skip all images before 100 if we are building the test/val set
            if not is_train and int(image_id) < 99:
                continue

            self.add_image("color_shapes", image_id=image_id, path=img_path)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """

        info = self.image_info[image_id]

        # ! Careful ! Not the same as the given image ID (probably due to training/val split?
        file_image_id = int(info.get("id"))

        img_path = info["path"]
        dataset_dir = str(Path(img_path).parent)
        scenes_dir = str(Path(img_path).parent.parent.joinpath("scenes"))

        filename = Path(img_path).name

        # Name of image file
        name = filename[:-4]

        scene_path = scenes_dir + "/" + name + ".json"

        with open(scene_path, "r") as f:
            json_data = json.load(f)

        txt_image_filename = "".join(listdir(dataset_dir))

        mask = np.zeros([320, 480, len(json_data["objects"])], dtype=np.uint8)

        class_ids = np.zeros(len(json_data["objects"]))

        for k in range(len(json_data["objects"])):
            # mask_path = images_dir + name + "_mask.png"
            # mask = read_image(mask_path)
            # obj_ids = torch.unique(mask)
            match = re.findall(r'CLEVR_new_%s_%d_mask_(\w+).png' % (str(file_image_id).zfill(6), k), txt_image_filename)
            classification = match[0]
            mask_filename = "CLEVR_new_%s_%d_mask_%s.png" % (str(file_image_id).zfill(6), k, classification)
            mask_path = dataset_dir + "/" + mask_filename
            raw_mask = imread(mask_path)

            for i in range(320):
                for j in range(480):
                    if raw_mask[i][j][0] == 255:
                        mask[i][j][k] = 1

            # We set the class id for the shape i
            class_ids[k] = self.class_names.index(classification)

        return mask, class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the color and shape data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "color_shapes":
            return info["color_shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)
