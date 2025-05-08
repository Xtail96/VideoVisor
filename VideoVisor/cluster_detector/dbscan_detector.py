import os.path
import cv2
import numpy as np
from typing import List
import utils
import colorsys
from sklearn.cluster import DBSCAN
import random
from tqdm import tqdm
from PIL import Image
from matplotlib import image


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


    def detect(self, img_path) -> (list, str):
        # resize img
        img = Image.open(img_path)
        w, h = img.width, img.height

        max_width, max_height = 512, 512
        target_size = (max_width, max_height * h / w) if w > h else (max_width * w / h, max_height)

        #img.resize((int(w * 0.25), int(h * 0.25))).save(f'{img_path}_resized.jpg')
        img.resize((int(target_size[0]), int(target_size[1]))).save(f'{img_path}_resized.jpg')
        img = image.imread(f'{img_path}_resized.jpg') / 255
        [m, n] = img.shape[:2]

        # NGRD - normalized green-red difference (NDVI inspired)
        f = (img[:, :, 1] - img[:, :, 0]) / (img[:, :, 1] + img[:, :, 0])
        f = 0.5 * (f + 1)

        k = 0
        x = np.zeros([m * n, 2])
        for i in range(m):
            for j in range(n):
                x[k, :] = [img[i, j, 2], f[i, j]]
                k = k + 1

        dbscan = DBSCAN(eps=0.0122, min_samples=500).fit(x)
        labels = dbscan.labels_
        #cluster_map = np.reshape(labels + 1, [m, n]) / 2
        cluster_map = np.reshape(labels, [m, n])
        cluster_labels = list(set(labels))

        detected_clusters = []
        for label in cluster_labels:
            coords = self.get_cluster_coords(cluster_map, label)
            detected_clusters.append(self.create_cluster(coords, int(os.path.basename(img_path).split('.')[0]), label))

        for cluster in detected_clusters:
            cluster.draw(f'{img_path}_resized.jpg', list(np.random.choice(range(256), size=3)))

        return detected_clusters, img_path

    @staticmethod
    def get_cluster_coords(segmented_image, class_index):
        coords = []
        y, x = segmented_image.shape
        for i in range(y):
            for j in range(x):
                if segmented_image[i, j] == class_index:
                    coords.append(utils.Point2D(j, i))
        return coords

    @staticmethod
    def create_cluster(coords, frame, label) -> utils.Cluster2D:
        return utils.Cluster2D(coords, label)

    def _detect(self, img_path: str) -> (List[utils.DetectedObject], str):
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
        labels = DBSCAN(eps=0.05, min_samples=9).fit_predict(pixel_values)
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

    def detect_all(self, images: List[str], target_classes: List[str], silent: bool) -> List[utils.DetectedObject]:
        return list(self.detect(img) for img in tqdm(images[::25], f'{type(self).__name__}'))
