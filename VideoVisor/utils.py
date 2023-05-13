import shutil
import os
import cv2
from typing import List


class BoundingBox:
    def __init__(self, x: float, y: float, w: float, h: float):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    def pack(self):
        return self.x, self.y, self.width, self.height


class DetectedObject:
    def __init__(self, label, bbox):
        self.label = label
        self.bbox = BoundingBox(*bbox)

    def to_string(self) -> str:
        return f'(label: {self.label}, x:{self.bbox.x}, y:{self.bbox.y}, w:{self.bbox.width}, h:{self.bbox.height})'


def get_video_frames(source_video_path, output_dir):
    video_frames_dir = os.path.join(output_dir, os.path.basename(source_video_path))
    frames = list(os.path.abspath(os.path.join(video_frames_dir, frame)) for frame in os.listdir(video_frames_dir))
    frames.sort(key=len)
    return frames


def draw_objects_on_image(objects: List[DetectedObject], image_path: str) -> None:
    image = cv2.imread(image_path)
    for item in objects:
        image = draw_bounding_box(image, item.label, item.bbox.pack())
    cv2.imwrite(image_path, image)


def draw_bounding_box(image, label, box):
    """
    Drawing object borders with captions
    :param image: original image
    :param label: text fo drawing near the bbox
    :param box: coordinates of the area around the object
    :return: image with marked objects
    """

    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image, start, end, color, width)
    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = label
    return cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)


def clear_folder(folder: str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
