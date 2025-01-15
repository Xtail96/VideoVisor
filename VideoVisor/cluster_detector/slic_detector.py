import os.path
import cv2
import numpy as np
from typing import List
import utils
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plte

class SLICDetector:
    def __init__(self):
        pass

    def create_segmented_image(self, classes_coords):
        pass

    @staticmethod
    def get_cluster_coords(segmented_image, class_index):
        coords = []
        x, y = segmented_image.shape
        for i in range(x):
            for j in range(y):
                if segmented_image[i, j] == class_index:
                    coords.append((j, i))
        return coords

    @staticmethod
    def get_cluster_center(coords):
        a = np.array(coords)
        mean = np.mean(a, axis=0)
        return mean[0], mean[1]

    @staticmethod
    def create_cluster_bbox(cluster_coords, frame, label) -> utils.DetectedObject:
        #cluster_coords = sorted(cluster_coords, key=lambda k: [k[0], k[1]])
        cluster_x = [k[0] for k in cluster_coords]
        cluster_y = [k[1] for k in cluster_coords]

        top_left = (min(cluster_x), min(cluster_y))
        right_bottom = (max(cluster_x), max(cluster_y))
        width = right_bottom[0] - top_left[0]
        height = right_bottom[1] - top_left[1]

        # Вычисление центра масс кластера и создание окрестности
        centroid_x, centroid_y = SLICDetector.get_cluster_center(cluster_coords)
        bbox_top_left_x = int(centroid_x - width / 10)
        bbox_top_left_y = int(centroid_y - height / 10)
        bbox_width = int(width / 10)
        bbox_height = int(height / 10)

        if width < 0 or height < 0:
            raise Exception('width and height can not be < 0')

        return utils.DetectedObject(label, [bbox_top_left_x, bbox_top_left_y, bbox_width, bbox_height], frame)

    def detect(self, img_path: str, debug=False) -> (List[utils.DetectedObject], str):
        print(f'Try to detect objects on {img_path}')
        image = img_as_float(io.imread(img_path))
        segmented_image = slic(image, n_segments=100)
        frame_number = int(os.path.basename(img_path).split('.')[0])
        detected_objects = []

        classes_count = segmented_image.max() - 1
        classes_coords = []
        for i in range(classes_count):
            coords = self.get_cluster_coords(segmented_image, i + 1)
            detected_objects.append(self.create_cluster_bbox(coords, frame_number, f'{i + 1}'))
            classes_coords.append(coords)

        utils.draw_objects_on_image(detected_objects, img_path)

        if debug:
            float_img = mark_boundaries(image, segmented_image)
            int_img = float_img * 1000
            cv2.imwrite(f'{img_path}_slic.jpg', int_img)
        return detected_objects, img_path

    def detect_all(self, images: List[str], target_classes: List[str], silent: bool) -> List[utils.DetectedObject]:
        return list(self.detect(img, True) for img in images[::25])
