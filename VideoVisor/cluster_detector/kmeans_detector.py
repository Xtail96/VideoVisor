import os.path
import cv2
import numpy as np
from typing import List
import utils


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

    def detect(self, img_path: str, target_classes: List[str]) -> (List[utils.DetectedObject], str):
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
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        compactness, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        #compactness, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        pixel_values, labels, centers = compactness, labels, np.uint8(centers)

        #segmented_image = self.create_segmented_image(source_image, labels, centers)
        segmented_image = self.create_segmented_image_rgb(source_image, labels)
        cv2.imwrite(f'{img_path}_kmeans.jpg', segmented_image)

        # utils.draw_objects_on_image(detected_objects, img_path)
        # print(f'{list(detected_object.to_string() for detected_object in detected_objects)} detected')
        return list(), img_path

    def detect_all(self, images: List[str], target_classes: List[str]) -> List[utils.DetectedObject]:
        return (list(self.detect(img, target_classes) for img in images[42:80]) +
                list(self.detect(img, target_classes) for img in images[138:170]))
        #return list(self.detect(img, target_classes) for img in images)
