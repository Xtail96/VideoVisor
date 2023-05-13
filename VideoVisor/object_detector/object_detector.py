import os.path

import cv2
import numpy as np
from typing import List


class ObjectDetector:
    def __init__(self):
        # Loading YOLO scales from files and setting up the network
        self.net = cv2.dnn.readNetFromDarknet(self.get_resource_path('yolov4-tiny.cfg'),
                                         self.get_resource_path('yolov4-tiny.weights'))
        layer_names = self.net.getLayerNames()
        self.out_layers = [layer_names[index - 1] for index in self.net.getUnconnectedOutLayers()]
        print(layer_names)
        print(self.out_layers)

        # Loading from a file of object classes that YOLO can detect
        with open(self.get_resource_path('coco.names.txt')) as file:
            self.classes = file.read().split('\n')
        print(self.classes)

    @staticmethod
    def get_resource_path(resource_name: str) -> str:
        return os.path.abspath(os.path.join('object_detector', 'Resources', resource_name))

    def detect(self, img_path: str) -> (List, str):
        image = cv2.imread(img_path)
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (608, 608),
                                     (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)
        class_indexes, class_scores, boxes = ([] for i in range(3))
        objects_count = 0

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
                    box = [center_x - obj_width // 2, center_y - obj_height // 2,
                           obj_width, obj_height]
                    boxes.append(box)
                    class_indexes.append(class_index)
                    class_scores.append(float(class_score))

        chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
        for box_index in chosen_boxes:
            box_index = box_index
            box = boxes[box_index]
            class_index = class_indexes[box_index]
            self.draw_object_bounding_box(image, self.classes[class_index], box)

        cv2.imwrite(img_path, image)
        return [], img_path

    @staticmethod
    def draw_object_bounding_box(image_to_process, label, box):
        """
        Drawing object borders with captions
        :param image_to_process: original image
        :param label
        :param box: coordinates of the area around the object
        :return: image with marked objects
        """

        x, y, w, h = box
        start = (x, y)
        end = (x + w, y + h)
        color = (0, 255, 0)
        width = 2
        final_image = cv2.rectangle(image_to_process, start, end, color, width)

        start = (x, y - 10)
        font_size = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        width = 2
        text = label
        final_image = cv2.putText(final_image, text, start, font,
                                  font_size, color, width, cv2.LINE_AA)


    def detect_all(self, images: List[str]) -> List:
        return list(self.detect(img) for img in images)


