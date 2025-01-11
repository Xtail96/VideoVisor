import os.path
import cv2
import numpy as np
from typing import List
import utils
import colorsys
from sklearn.cluster import DBSCAN
import random


class DBSCANDetector:
    def __init__(self):
        pass

    #def assign_colors_2_clusters(self, clusters):
    #    colored = np.array([[x, x, x] for x in clusters])
    #    return colored

    def assign_colors_2_clusters(self, labels):
        cluster_keys = list(set(labels))
        cluster_values = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for x in cluster_keys]
        cluster_dict = dict(zip(cluster_keys, cluster_values))
        colored = np.array([cluster_dict[x] for x in labels])
        return colored

    def detect(self, img_path: str) -> (List[utils.DetectedObject], str):
        print(f'Try to detect objects on {img_path}')
        source_image = cv2.imread(img_path)
        height, width, _ = source_image.shape
        #image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        image = source_image
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        pixel_values = np.array([x / 1000 for x in pixel_values])

        #clustering = DBSCAN(eps=0.3, min_samples=10).fit(pixel_values)
        #clustering = DBSCAN(eps=2, min_samples=100).fit(pixel_values)
        #clustering = DBSCAN(eps=1.731, min_samples=1000).fit(pixel_values)
        #clustering = DBSCAN(eps=1, min_samples=250).fit(pixel_values)
        #print(f'clusters min: {min(clustering.labels_)}, max: {max(clustering.labels_)}')
        #colored = self.assign_colors_2_clusters(clustering.labels_)

        #labels = DBSCAN(eps=1.42, min_samples=2500).fit_predict(pixel_values)
        labels = DBSCAN(eps=0.05, min_samples=2500).fit_predict(pixel_values)
        print(f'clusters min: {min(labels)}, max: {max(labels)}')
        colored = self.assign_colors_2_clusters(labels)
        colored_mask = colored.reshape(height, width, 3)
        cv2.imwrite(f'{img_path}_dbscan.jpg', colored_mask)

        #mask = clustering.labels_.reshape(height, width, 1)
        mask = labels.reshape(height, width, 1)
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
        return (list(self.detect(img) for img in images[74:75]) +
                list(self.detect(img) for img in images[141:143]) +
                list(self.detect(img) for img in images[277:278]))
        #return (list(self.detect(img) for img in images[42:80]) +
        #        list(self.detect(img) for img in images[138:170]))
        #return list(self.detect(img) for img in images[70:75])
        #return list(self.detect(img) for img in images)
