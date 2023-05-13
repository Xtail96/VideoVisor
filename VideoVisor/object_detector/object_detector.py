import os.path
import cv2
import numpy as np
from typing import List
import utils


class ObjectDetector:
    def __init__(self):
        self.model = cv2.dnn.readNetFromDarknet(self.get_resource_path('yolov4-tiny.cfg'),
                                                self.get_resource_path('yolov4-tiny.weights'))
        self.out_layers = [self.model.getLayerNames()[index - 1] for index in self.model.getUnconnectedOutLayers()]
        with open(self.get_resource_path('coco.names.txt')) as file:
            self.classes = file.read().split('\n')

    @staticmethod
    def get_resource_path(resource_name: str) -> str:
        return os.path.abspath(os.path.join('object_detector', 'Resources', resource_name))

    def detect(self, img_path: str, target_classes: List[str] = []) -> (List[utils.DetectedObject], str):
        print(f'Try to detect objects on {img_path}')
        image = cv2.imread(img_path)
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (608, 608),
                                     (0, 0, 0), swapRB=True, crop=False)
        self.model.setInput(blob)
        outs = self.model.forward(self.out_layers)
        class_indexes, class_scores, boxes = ([] for i in range(3))

        # Starting a search for objects in an image
        for out in outs:
            for obj in out:
                scores = obj[5:]
                class_index = np.argmax(scores)
                class_score = scores[class_index]
                if class_score > 0:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    obj_width = int(obj[2] * width)
                    obj_height = int(obj[3] * height)
                    boxes.append([center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height])
                    class_indexes.append(class_index)
                    class_scores.append(float(class_score))
        chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)

        detected_objects = []
        for box_index in chosen_boxes:
            box_index = box_index
            box = boxes[box_index]
            class_index = class_indexes[box_index]
            if not target_classes or self.classes[class_index] in target_classes:
                detected_objects.append(utils.DetectedObject(self.classes[class_index], box))

        utils.draw_objects_on_image(detected_objects, img_path)
        print(f'{list(detected_object.to_string() for detected_object in detected_objects)} detected')
        return detected_objects, img_path

    def detect_all(self, images: List[str], target_classes: List[str] = []) -> List[utils.DetectedObject]:
        return list(self.detect(img, target_classes) for img in images)
