import cv2
import numpy as np
from typing import List


class ObjectDetector:
    def __init__(self):
        pass

    def detect(self, image: str) -> (List, str):
        pass

    def detect_all(self, images: List[str]) -> List:
        return list(self.detect(img) for img in images)


