import os

from keras_preprocessing.image import load_img, img_to_array

import mrcnn.model as modellib
from mrcnn import visualize
from samples.clevr.KnowledgeBase import KnowledgeBase
from samples.clevr.Query import Query
from samples.clevr.colors.clevr_colors import CLEVRColorConfig
from samples.clevr.shapes.clevr_shapes import CLEVRShapeConfig

ROOT_DIR = os.path.abspath("../../")

# Directory to save logs and trained model
SHAPE_MODEL_DIR = os.path.join(ROOT_DIR, "samples/clevr/shapes/logs")
COLOR_MODEL_DIR = os.path.join(ROOT_DIR, "samples/clevr/colors/logs")


class ShapeInferenceConfig(CLEVRShapeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


shape_inference_config = ShapeInferenceConfig()


class ColorInferenceConfig(CLEVRColorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


color_inference_config = ColorInferenceConfig()

# Recreate the model in inference mode
shape_model = modellib.MaskRCNN(mode="inference", config=shape_inference_config, model_dir=SHAPE_MODEL_DIR)
color_model = modellib.MaskRCNN(mode="inference", config=color_inference_config, model_dir=COLOR_MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# shape_model_path = os.path.join(ROOT_DIR, "clevr_shape_mask_rcnn_coco.h5")
# color_model_path = os.path.join(ROOT_DIR, "clevr_color_mask_rcnn_coco.h5")
shape_model_path = shape_model.find_last()
color_model_path = color_model.find_last()

# Load trained weights
print("Loading weights from ", shape_model_path)
shape_model.load_weights(shape_model_path, by_name=True)

# print("Loading weights from ", color_model_path)
# color_model.load_weights(color_model_path, by_name=True)

img = load_img("CLEVR_new_000089.png")
shape_image = img_to_array(img)
# color_image = img_to_array(img)

img.show()

# detecting objects in the image
shape_results = shape_model.detect([shape_image])
# color_results = color_model.detect([color_image])

shape_result = shape_results[0]

visualize.display_instances(shape_image, shape_result['rois'], shape_result['masks'], shape_result['class_ids'],
                            ["", "cube", "sphere", "cylinder"], shape_result['scores'])

# color_result = color_results[0]

# sentence = "select all green cubes and spheres left of the cylinder"
# sentence = "select all purple cubes left of the cylinders"
# sentence = "select the cyan cylinder on the right of the cube"
sentence = "select the cylinder right of the purple cube and left of the cyan cylinder"
# sentence = "select all cyan and gray cylinders"
query = Query(sentence)

knowledge_base = KnowledgeBase()
knowledge_base.update("shape", shape_result)
# knowledge_base.update("color", color_image)
# knowledge_base.update("color", color_result)
knowledge_base.reason(query)
