from typing import Dict, Set, List

from numpy import ndarray

from samples.clevr.Query import Query


class KnowledgeBase:

    def __init__(self):
        # A dictionary keeps knowledge about each knowledge
        self.knowledge = dict()

    def update(self, feature: str, results: Dict[str, ndarray]):
        color, shape = feature.split("_")
        class_ids = results["class_ids"]
        rois = results["rois"]

        nr_objects = len(class_ids)

        class_names = ["cube", "sphere", "cylinder"]

        for k in range(nr_objects):
            roi = rois[k]
            shape = class_ids[k]
            y0 = roi[0]
            x0 = roi[1]
            y1 = roi[2]
            x1 = roi[3]

            # We look at x_center to reason about "left_of"
            x_center = min(x0, x1) + abs(x1 - x0) / 2

            # We look at y_bottom to reason about "in_front_of"
            y_bottom = min(y0, y1)

            shape_name = class_names[shape - 1]

            self.knowledge[k] = dict()
            self.knowledge[k]["x_center"] = x_center
            self.knowledge[k]["y_bottom"] = y_bottom
            self.knowledge[k]["color"] = color
            self.knowledge[k]["shape"] = shape_name

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

    def _object_id_to_pixel_coordinates(self, object_id: int):
        pass

    def reason(self, query: Query):
        action_and_object_ids = []
        for (action, filters) in query.get_filters_by_actions():
            object_sets = []
            for (shape, object_filters) in filters:
                object_ids = self._get_object_ids_by_shape(shape)
                for object_filter in object_filters.keys():
                    if object_filter == "color":
                        object_ids = self._filter_object_ids_by_colors(object_filters[object_filter], object_ids)
                    object_sets.append(object_ids)

            new_object_sets = []
            for i in range(len(filters)):
                object_set = object_sets[i]
                object_filters = filters[i][1]
                for object_filter in object_filters:
                    if object_filter == "relation":
                        new_object_ids = self._filter_object_ids_by_relations(object_filters[object_filter], object_set, object_sets)
                        new_object_sets.append(new_object_ids)

            action_and_object_ids.append((action, object_ids))

        action_and_pixel_coordinates = []
        for (action, object_ids) in action_and_object_ids:

            pixel_coordinates = []
            for object_id in object_ids:
                pixel_coordinates.append(self._object_id_to_pixel_coordinates(object_id))
        return action_and_pixel_coordinates


