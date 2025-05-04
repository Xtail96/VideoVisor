import os.path
import cv2
import numpy as np
from typing import List
import utils
from ultralytics import YOLO
from tqdm import tqdm


class YOLO11Detector:
    def __init__(self):
        # Load a model
        self.model = YOLO("yolo11n.pt")

    def detect(self, img_path: str, target_classes: List[str], silent: bool) -> (List[utils.DetectedObject], str):
        if not silent:
            print(f'Try to detect objects on {img_path}')

        results = self.model(img_path)
        # Process results list
        detected_objects = []
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            #result.show()  # display to screen
            result.save(filename=f'{img_path}_y11.jpg')  # save to disk

            frame_number = int(os.path.basename(img_path).split('.')[0])
            for box in boxes:
                x = box.xywh.data[0][0].item()
                y = box.xywh.data[0][1].item()
                w = box.xywh.data[0][2].item()
                h = box.xywh.data[0][3].item()

                # Координаты центра переведем в координаты левого верхнего угла
                x = int(x - w / 2)
                y = int(y - h / 2)
                w = int(w)
                h = int(h)

                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                if class_name in target_classes:
                    detected_objects.append(utils.DetectedObject(class_name, [x, y, w, h], frame_number))

        utils.draw_objects_on_image(detected_objects, img_path)
        if not silent:
            print(f'{list(detected_object.to_string() for detected_object in detected_objects)} detected')

        return list(x for x in detected_objects), img_path

    def detect_all(self, images: List[str], target_classes: List[str], silent: bool) -> List[utils.DetectedObject]:
        #return list(self.detect(img, target_classes, silent) for img in images[75:80])
        return list(self.detect(img, target_classes, silent) for img in tqdm(images[::25], f'{type(self).__name__}'))
