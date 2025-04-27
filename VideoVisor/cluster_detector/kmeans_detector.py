import os.path
import cv2
import numpy as np
from typing import List
import utils
import colorsys
from tqdm import tqdm


class KMeansDetector:
    def __init__(self):
        pass

    def create_segmented_image(self, image, labels, centers):
        """Create a segmented image using the cluster centroids."""
        segmented_image = centers[labels.flatten()]
        return segmented_image.reshape(image.shape)

    @staticmethod
    def get_cluster_coords_alt(segmented_image, r, g, b) -> List[utils.Point2D]:
        coords = [[utils.Point2D(x, y) for x in range(len(segmented_image[y]))
                   if segmented_image[y, x, 0] == b and segmented_image[y, x, 1] == g and segmented_image[y, x, 2] == r]
                  for y in range(len(segmented_image))]
        coords = [x for row in coords for x in row]
        return coords

    @staticmethod
    def get_cluster_coords(segmented_image, r, g, b) -> List[utils.Point2D]:
        coords = []
        x, y, z = segmented_image.shape
        for i in range(x):
            for j in range(y):
                if segmented_image[i, j, 0] == b and segmented_image[i, j, 1] == g and segmented_image[i, j, 2] == r:
                    coords.append(utils.Point2D(j, i))
        return coords

    @staticmethod
    def create_cluster(coords, frame, label) -> utils.Cluster2D:
        return utils.Cluster2D(coords, label)

    def detect(self, img_path: str, debug=False) -> (List[utils.DetectedObject], str):
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
        frame_number = int(os.path.basename(img_path).split('.')[0])
        detected_clusters = []
        i = 0
        for center in centers:
            coords = self.get_cluster_coords_alt(segmented_image, center[2], center[1], center[0])
            detected_clusters.append(
                {
                    'cluster': self.create_cluster(coords, frame_number, f'{i}'),
                    'color': (center[2], center[1], center[0])
                }
            )
            i = i + 1

        if debug:
            for cluster in detected_clusters:
                cluster['cluster'].draw(img_path, cluster['color'])
        return list([x['cluster'] for x in detected_clusters]), img_path

    def detect_all(self, images: List[str], target_classes: List[str], silent: bool) -> List[utils.DetectedObject]:
        #return (list(self.detect(img) for img in images[42:80]) + list(self.detect(img) for img in images[138:170]))
        return list(self.detect(img, not silent) for img in tqdm(images[::100], f'{type(self).__name__}'))
