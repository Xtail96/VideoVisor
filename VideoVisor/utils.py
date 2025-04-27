import shutil
import os
import cv2
from typing import List


class Point2D:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __hash__(self):
        return hash(f'{self.x}{self.y}')

    def draw_on_image(self, image, color=(0, 0, 255)):
        return cv2.circle(image, (self.x, self.y), radius=0, color=color, thickness=-1)


class Cluster2D:
    def __init__(self, points: List[Point2D], label: str):
        self.points = points
        self.label = label

    def intersection(self, other):
        intersection = list(set(self.points) & set(other.points))
        return len(intersection), intersection

    def union(self, other):
        joined = self.points + other.points
        union = list(set(joined))
        return len(union), union

    def draw(self, image_path, color=(0, 0, 255)):
        color = (int(color[0]), int(color[1]), int(color[2]))

        image = cv2.imread(image_path)
        for point in self.points:
            image = point.draw_on_image(image, color)
        cv2.imwrite(image_path, image)

    def intersects_with(self, other, iou_treshold=0.75):
        return self.intersection(other)[0] / self.union(other)[0] >= iou_treshold


class BoundingBox:
    def __init__(self, x: float, y: float, w: float, h: float, frame: int):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.frame = frame

    def control_points(self):
        return (self.x, self.y), \
            (self.x + self.width, self.y + self.height)

    def intersects_with(self, other, iou_treshold=0.75) -> bool:
        if self.frame != other.frame:
            return False

        intersection = self.get_intersection_square(other)
        union = self.get_union_square(other)
        if intersection <= 0 or union <= 0:
            return False
        iou = intersection / union
        return iou >= iou_treshold

    def get_union_square(self, other) -> float:
        if self.frame != other.frame:
            return 0

        my_control_points = self.control_points()
        other_control_points = other.control_points()

        union_left = min(my_control_points[0][0], other_control_points[0][0])
        union_bottom = max(my_control_points[1][1], other_control_points[1][1])
        union_right = max(my_control_points[1][0], other_control_points[1][0])
        union_top = min(my_control_points[0][1], other_control_points[0][1])

        union_width = union_right - union_left
        union_height = union_bottom - union_top
        intersection_square = union_width * union_height
        return intersection_square if intersection_square > 0 else 0

    def get_intersection_square(self, other) -> float:
        if self.frame != other.frame:
            return 0

        my_control_points = self.control_points()
        other_control_points = other.control_points()

        intersection_left = max(my_control_points[0][0], other_control_points[0][0])
        intersection_bottom = min(my_control_points[1][1], other_control_points[1][1])
        intersection_right = min(my_control_points[1][0], other_control_points[1][0])
        intersection_top = max(my_control_points[0][1], other_control_points[0][1])

        intersection_width = intersection_right - intersection_left
        intersection_height = intersection_bottom - intersection_top
        intersection_square = intersection_width * intersection_height
        return intersection_square if intersection_square > 0 else 0

    def pack(self):
        return self.x, self.y, self.width, self.height


class DetectedObject:
    def __init__(self, label, bbox, frame):
        self.label = label
        bbox.append(frame)
        self.bbox = BoundingBox(*bbox)

    def intersects_with(self, other, iou_treshold=0.75):
        #if self.label != other.label:
        #    return False
        return self.bbox.intersects_with(other.bbox, iou_treshold)

    def to_string(self) -> str:
        return f'(label: {self.label}, x:{self.bbox.x}, y:{self.bbox.y}, w:{self.bbox.width}, h:{self.bbox.height})'


def get_video_frames(source_video_path, output_dir) -> List[str]:
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


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def copy_file(src, dst):
    return shutil.copy(src, dst)


def create_folder(dst):
    os.mkdir(dst)
