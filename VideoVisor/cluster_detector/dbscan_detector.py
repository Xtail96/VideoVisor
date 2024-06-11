import os.path
import cv2
import numpy as np
from typing import List
import utils
import colorsys
from sklearn.cluster import DBSCAN


class DBSCANDetector:
    def __init__(self):
        pass

    def detect(self, img_path: str) -> (List[utils.DetectedObject], str):
        print(f'Try to detect objects on {img_path}')

        source_image = cv2.imread(img_path)
        height, width, _ = source_image.shape
        #image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        image = source_image
        pixel_values = image.reshape((-1, 3))

        #clustering = DBSCAN(eps=0.3, min_samples=10).fit(pixel_values)
        #clustering = DBSCAN(eps=2, min_samples=100).fit(pixel_values)
        clustering = DBSCAN(eps=2, min_samples=100).fit(pixel_values)
        colored = np.array([[x, x, x] for x in clustering.labels_])
        colored_mask = colored.reshape(height, width, 3)
        cv2.imwrite(f'{img_path}_dbscan.jpg', colored_mask)

        mask = clustering.labels_.reshape(height, width, 1)
        tmp = []
        for i in range(height):
            row = []
            for j in range(width):
                row.append(source_image[i][j] if mask[i][j] != -1 else [255, 0, 0])
            tmp.append(row)
        tmp = np.array(tmp)
        cv2.imwrite(f'{img_path}_dbscan_denoised.jpg', tmp)

        frame_number = int(os.path.basename(img_path).split('.')[0])
        detected_objects = []
        i = 0

        #utils.draw_objects_on_image(detected_objects, img_path)
        # print(f'{list(detected_object.to_string() for detected_object in detected_objects)} detected')
        return detected_objects, img_path

    def detect_all(self, images: List[str]) -> List[utils.DetectedObject]:
        return (list(self.detect(img) for img in images[42:80]) +
                list(self.detect(img) for img in images[138:170]))
        #return list(self.detect(img) for img in images)
