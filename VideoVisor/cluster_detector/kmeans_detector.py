import os.path
import cv2
import numpy as np
from typing import List
import utils
import colorsys


class KMeansDetector:
    def __init__(self):
        pass

    def create_segmented_image(self, image, labels, centers):
        """Create a segmented image using the cluster centroids."""
        segmented_image = centers[labels.flatten()]
        return segmented_image.reshape(image.shape)

    def create_masked_image(self, image, labels, cluster_to_disable):
        """Create a masked image by disabling a specific cluster."""
        masked_image = np.copy(image).reshape((-1, 3))
        masked_image[labels.flatten() == cluster_to_disable] = [0, 0, 0]
        return masked_image.reshape(image.shape)

    def create_segmented_image_rgb(self, image, labels):
        """Create a masked image by disabling a specific cluster."""
        masked_image = np.copy(image).reshape((-1, 3))
        masked_image[labels.flatten() == 0] = [128, 0, 0]
        masked_image[labels.flatten() == 1] = [0, 128, 0]
        masked_image[labels.flatten() == 2] = [0, 0, 128]
        return masked_image.reshape(image.shape)

    @staticmethod
    def get_cluster_coords(segmented_image, r, g, b):
        coords = []
        x, y, z = segmented_image.shape
        for i in range(x):
            for j in range(y):
                if segmented_image[i, j, 0] == b and segmented_image[i, j, 1] == g and segmented_image[i, j, 2] == r:
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
        centroid_x, centroid_y = KMeansDetector.get_cluster_center(cluster_coords)
        bbox_top_left_x = int(centroid_x - width / 10)
        bbox_top_left_y = int(centroid_y - height / 10)
        bbox_width = int(width / 10)
        bbox_height = int(height / 10)

        if width < 0 or height < 0:
            raise Exception('width and height can not be < 0')

        return utils.DetectedObject(label, [bbox_top_left_x, bbox_top_left_y, bbox_width, bbox_height], frame)

    def detect(self, img_path: str) -> (List[utils.DetectedObject], str):
        print(f'Try to detect objects on {img_path}')

        """Read the image and convert it to RGB."""
        source_image = cv2.imread(img_path)
        height, width, _ = source_image.shape
        image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        """Reshape the image to a 2D array of pixels and 3 color values (RGB) and convert to float."""
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        """Perform k-means clustering on the pixel values."""
        k = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        #compactness, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        compactness, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        pixel_values, labels, centers = compactness, labels, np.uint8(centers)

        segmented_image = self.create_segmented_image(source_image, labels, centers)
        cv2.imwrite(f'{img_path}_kmeans.jpg', segmented_image)
        segmented_image_rgb = self.create_segmented_image_rgb(source_image, labels)
        cv2.imwrite(f'{img_path}_kmeans_rgb.jpg', segmented_image_rgb)

        frame_number = int(os.path.basename(img_path).split('.')[0])
        detected_objects = []
        i = 0
        for center in centers:
            coords = self.get_cluster_coords(segmented_image, center[2], center[1], center[0])
            detected_objects.append(self.create_cluster_bbox(coords, frame_number, f'{i}'))
            i = i + 1

        utils.draw_objects_on_image(detected_objects, img_path)
        # print(f'{list(detected_object.to_string() for detected_object in detected_objects)} detected')
        return detected_objects, img_path

    def detect_all(self, images: List[str], target_classes: List[str], silent: bool) -> List[utils.DetectedObject]:
        #return (list(self.detect(img) for img in images[42:80]) + list(self.detect(img) for img in images[138:170]))
        return list(self.detect(img) for img in images)
