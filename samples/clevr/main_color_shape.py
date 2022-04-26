import os

from keras_preprocessing.image import load_img, img_to_array

import mrcnn.model as modellib
from mrcnn import visualize
from samples.clevr.KnowledgeBase import KnowledgeBase
from samples.clevr.Query import Query
from samples.clevr.color_shapes.clevr_color_shapes import CLEVRColorShapeConfig

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

ROOT_DIR = os.path.abspath("../../")

# Directory to save logs and trained model
COLOR_SHAPE_MODEL_DIR = os.path.join(ROOT_DIR, "samples/clevr/color_shapes/medium/logs")


class ColorShapeInferenceConfig(CLEVRColorShapeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


color_shape_inference_config = ColorShapeInferenceConfig()

# Recreate the model in inference mode
color_shape_model = modellib.MaskRCNN(mode="inference", config=color_shape_inference_config, model_dir=COLOR_SHAPE_MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# shape_model_path = os.path.join(ROOT_DIR, "clevr_shape_mask_rcnn_coco.h5")
# color_model_path = os.path.join(ROOT_DIR, "clevr_color_mask_rcnn_coco.h5")
color_shape_model_path = color_shape_model.find_last()

# Load trained weights
print("Loading weights from ", color_shape_model_path)
color_shape_model.load_weights(color_shape_model_path, by_name=True)

# print("Loading weights from ", color_model_path)
# color_model.load_weights(color_model_path, by_name=True)

img = load_img("../images/CLEVR_new_000000.png")
color_shape_image = img_to_array(img)
# color_image = img_to_array(img)

img.show()

# detecting objects in the image
color_shape_results = color_shape_model.detect([color_shape_image])
# color_results = color_model.detect([color_image])

color_shape_result = color_shape_results[0]

visualize.display_instances(color_shape_image,
                            color_shape_result['rois'],
                            color_shape_result['masks'],
                            color_shape_result['class_ids'],
                            ["", "red_cube", "red_sphere", "red_cylinder",
                             "blue_cube", "blue_sphere", "blue_cylinder",
                             "green_cube", "green_sphere", "green_cylinder"],
                            color_shape_result['scores'])

# color_result = color_results[0]

# sentence = "select all green cubes and spheres left of the cylinder"
# sentence = "select all purple cubes left of the cylinders"
# sentence = "select the cyan cylinder on the right of the cube"
# sentence = "select the cylinder right of the purple cube and left of the cyan cylinder"
# sentence = "select the green cube right of the blue spheres"
# sentence = "select the blue spheres left of the green cube"
# sentence = "select all cyan and gray cylinders"
# sentence = "select all objects"
sentence = "select all blue cylinders"
query = Query(sentence)

knowledge_base = KnowledgeBase()
knowledge_base.update(color_shape_result)
action_and_centers = knowledge_base.reason(query)
print(action_and_centers)
