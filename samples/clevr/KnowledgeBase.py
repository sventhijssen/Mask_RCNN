from typing import Dict, Set, List

import numpy as np
from numpy import ndarray

from samples.clevr.Query import Query


class KnowledgeBase:

    def __init__(self):
        # A dictionary keeps knowledge about each knowledge
        self.knowledge = dict()

    def update(self, results: Dict[str, ndarray]):
        class_ids = results["class_ids"]
        rois = results["rois"]
        masks = results["masks"]

        class_names = ["", "red_cube", "red_sphere", "red_cylinder",
         "blue_cube", "blue_sphere", "blue_cylinder",
         "green_cube", "green_sphere", "green_cylinder"]

        nr_objects = len(class_ids)

        for k in range(nr_objects):
            roi = rois[k]

            mask = np.zeros((320, 480))
            for x in range(480):
                for y in range(320):
                    mask[y][x] = masks[y][x][k]

            # mask = masks[k]
            class_id = int(class_ids[k])
            class_name = class_names[class_id]
            color_name, shape_name = class_name.split("_")

            y0 = roi[0]
            x0 = roi[1]
            y1 = roi[2]
            x1 = roi[3]

            # We look at x_center to reason about "left_of"
            x_center = min(x0, x1) + abs(x1 - x0) / 2

            # We look at y_bottom to reason about "in_front_of"
            y_bottom = min(y0, y1)

            y_center = min(y0, y1) + abs(y1 - y0) / 2

            self.knowledge[k] = dict()
            self.knowledge[k]["x_center"] = x_center
            self.knowledge[k]["y_center"] = y_center
            self.knowledge[k]["y_bottom"] = y_bottom
            self.knowledge[k]["color"] = color_name
            self.knowledge[k]["shape"] = shape_name
            self.knowledge[k]["mask"] = mask
            self.knowledge[k]["roi"] = rois[k]

    def _get_object_ids_by_shape(self, shape: str):
        object_ids = set()
        for (i, attributes) in self.knowledge.items():
            if shape == "object" or shape == "thing" or shape == "shape":
                object_ids.add(i)
            elif attributes["shape"] == shape:
                object_ids.add(i)
        return object_ids

    def _filter_object_ids_by_colors(self, colors: List[str], object_ids: Set[int]):
        filtered_object_ids = set()
        for object_id in object_ids:
            if self.knowledge[object_id]["color"] in colors or self.knowledge[object_id]["color"] == "FIX_COLOR":
                filtered_object_ids.add(object_id)
        return filtered_object_ids

    def _remove_object_id_by_left(self, object_id: int, other_object_set: int, object_sets: List[Set[int]]):
        for other_object_id in object_sets[other_object_set]:
            if self.knowledge[object_id]["x_center"] > self.knowledge[other_object_id]["x_center"]:
                return True
        return False

    def _remove_object_id_by_right(self, object_id: int, other_object_set: int, object_sets: List[Set[int]]):
        for other_object_id in object_sets[other_object_set]:
            if self.knowledge[object_id]["x_center"] < self.knowledge[other_object_id]["x_center"]:
                return True
        return False

    def _remove_object_id_by_front(self, object_id: int, other_object_set: int, object_sets: List[Set[int]]):
        for other_object_id in object_sets[other_object_set]:
            if self.knowledge[object_id]["y_bottom"] > self.knowledge[other_object_id]["y_bottom"]:
                return True
        return False

    def _remove_object_id_by_behind(self, object_id: int, other_object_set: int, object_sets: List[Set[int]]):
        for other_object_id in object_sets[other_object_set]:
            if self.knowledge[object_id]["y_bottom"] < self.knowledge[other_object_id]["y_bottom"]:
                return True
        return False

    def _filter_object_ids_by_relations(self, relations: List[str], object_ids: Set[int], object_sets: List[Set[int]]):
        remove_object_ids = set()
        for object_id in object_ids:
            for (relation, other_object_set) in relations:
                if relation == "left" and self._remove_object_id_by_left(object_id, other_object_set, object_sets):
                    remove_object_ids.add(object_id)
                elif relation == "right" and self._remove_object_id_by_right(object_id, other_object_set, object_sets):
                    remove_object_ids.add(object_id)
                elif relation == "front" and self._remove_object_id_by_front(object_id, other_object_set, object_sets):
                    remove_object_ids.add(object_id)
                elif relation == "behind" and self._remove_object_id_by_behind(object_id, other_object_set, object_sets):
                    remove_object_ids.add(object_id)
        return object_ids - remove_object_ids

    def _object_id_to_center(self, object_id: int):
        return self.knowledge[object_id]["x_center"], self.knowledge[object_id]["y_center"]

    def reason(self, query: Query):
        action_and_object_ids = []
        for (action, filters) in query.get_filters_by_actions():
            object_sets = []
            for (shape, object_filters) in filters:
                print("Shape: {}, filters: {}".format(shape, object_filters))
                object_ids = self._get_object_ids_by_shape(shape)
                has_color_filter = False
                for object_filter in object_filters.keys():
                    if object_filter == "color":
                        has_color_filter = True
                        object_ids = self._filter_object_ids_by_colors(object_filters[object_filter], object_ids)
                        object_sets.append(object_ids)

                if not has_color_filter:
                    object_sets.append(object_ids)

            new_object_sets = []
            for i in range(len(filters)):
                object_set = object_sets[i]
                object_filters = filters[i][1]
                has_relation_filter = False
                for object_filter in object_filters:
                    if object_filter == "relation":
                        has_relation_filter = True
                        new_object_ids = self._filter_object_ids_by_relations(object_filters[object_filter], object_set, object_sets)
                        new_object_sets.append(new_object_ids)

                if not has_relation_filter:
                    new_object_sets.append(object_set)

            action_and_object_ids.append((action, new_object_sets))

        action_and_centers = []
        for (action, object_sets) in action_and_object_ids:
            centers = []
            for object_set in object_sets:
                for object_id in object_set:
                    centers.append(self._object_id_to_center(object_id))
            action_and_centers.append((action, centers))
        return action_and_centers
