import os
import re

from keras_preprocessing.image import load_img, img_to_array

import mrcnn.model as modellib
from samples.clevr.KnowledgeBase import KnowledgeBase
from samples.clevr.Query import Query

from samples.clevr.color_shapes.clevr_color_shapes import CLEVRColorShapeConfig


class Reasoner:

    def __init__(self):
        pass

    def _load_query(self, filepath='query.txt'):
        with open(filepath) as f:
            sentence = f.readlines()[0]
        sentence = re.sub(r"[^a-zA-Z0-9 ]", "", sentence)

        query = Query(sentence)

        return query

    def _load_image(self, filepath='capture.png'):
        img = load_img(filepath)
        return img_to_array(img)

    def reason(self):
        ROOT_DIR = os.path.abspath("../../")

        # Directory to save logs and trained model
        COLOR_SHAPE_MODEL_DIR = os.path.join(ROOT_DIR, "samples/clevr/color_shapes/medium/logs")

        class ColorShapeInferenceConfig(CLEVRColorShapeConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        color_shape_inference_config = ColorShapeInferenceConfig()

        # Recreate the model in inference mode
        color_shape_model = modellib.MaskRCNN(mode="inference", config=color_shape_inference_config,
                                              model_dir=COLOR_SHAPE_MODEL_DIR)

        # Get path to saved weights
        # Either set a specific path or find last trained weights
        color_shape_model_path = color_shape_model.find_last()

        # Load trained weights
        print("Loading weights from ", color_shape_model_path)
        color_shape_model.load_weights(color_shape_model_path, by_name=True)

        query = self._load_query()
        color_shape_image = self._load_image()

        # detecting objects in the image
        color_shape_results = color_shape_model.detect([color_shape_image])

        color_shape_result = color_shape_results[0]

        knowledge_base = KnowledgeBase()
        knowledge_base.update(color_shape_result)
        action_and_masks = knowledge_base.reason(query)

        return action_and_masks
