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
    def get_cluster_coords_(segmented_image, cluster_color_low, cluster_color_upper):
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2HSV)

        # hsv_cluster_color_low = colorsys.rgb_to_hsv(cluster_color_low[0], cluster_color_low[1], cluster_color_low[2])
        # hsv_cluster_color_upper = colorsys.rgb_to_hsv(cluster_color_upper[0], cluster_color_upper[1], cluster_color_upper[2])

        hsv_cluster_color_low = (0, 100, 50)
        hsv_cluster_color_upper = (0, 100, 50)

        # Define the lower and upper bounds of the color in HSV
        lower_color = np.array(hsv_cluster_color_low, dtype=np.uint8)
        upper_color = np.array(hsv_cluster_color_upper, dtype=np.uint8)

        # Threshold the image to get only the desired color
        mask = cv2.inRange(hsv_image, lower_color, upper_color)

        # Find the contours of the color regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through each contour and get the centroid of the region
        coordinates = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                coordinates.append((centroid_x, centroid_y))
        return coordinates

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
    def create_cluster_bbox(cluster_coords, frame, label) -> utils.DetectedObject:
        #cluster_coords = sorted(cluster_coords, key=lambda k: [k[0], k[1]])
        cluster_x = [k[0] for k in cluster_coords]
        cluster_y = [k[1] for k in cluster_coords]

        top_left = (min(cluster_x), min(cluster_y))
        right_bottom = (max(cluster_x), max(cluster_y))
        width = right_bottom[0] - top_left[0]
        height = right_bottom[1] - top_left[1]

        if width < 0 or height < 0:
            raise Exception('width and height can not be < 0')

        return utils.DetectedObject(label, [top_left[0], top_left[1], width, height], frame)


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
        compactness, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # compactness, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        pixel_values, labels, centers = compactness, labels, np.uint8(centers)

        #segmented_image = self.create_segmented_image(source_image, labels, centers)
        segmented_image = self.create_segmented_image_rgb(source_image, labels)
        cv2.imwrite(f'{img_path}_kmeans.jpg', segmented_image)

        frame_number = int(os.path.basename(img_path).split('.')[0])

        coords_1 = self.get_cluster_coords(segmented_image, 128, 0, 0)
        coords_2 = self.get_cluster_coords(segmented_image, 0, 128, 0)
        coords_3 = self.get_cluster_coords(segmented_image, 0, 0, 128)

        detected_objects = [self.create_cluster_bbox(coords_1, frame_number, '1'),
                            self.create_cluster_bbox(coords_2, frame_number, '2'),
                            self.create_cluster_bbox(coords_3, frame_number, '3')]
        utils.draw_objects_on_image(detected_objects, img_path)
        # print(f'{list(detected_object.to_string() for detected_object in detected_objects)} detected')
        return detected_objects, img_path

    def detect_all(self, images: List[str]) -> List[utils.DetectedObject]:
        return (list(self.detect(img) for img in images[42:80]) +
                list(self.detect(img) for img in images[138:170]))
        #return list(self.detect(img) for img in images)
