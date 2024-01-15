from video_parser.video_parser import VideoParser
import argparse
import os
import utils
from object_detector.object_detector import ObjectDetector
from typing import List
from noise_generator.noise_genertor import NoiseGenerator


def frame_f1(frame1: List[utils.DetectedObject], frame2: List[utils.DetectedObject]):
    if len(frame1) == 0 and len(frame2) == 0:
        return None

    true_positives = 0
    false_negatives = 0
    for x in frame1:
        found = False
        for y in frame2:
            if x.intersects_with(y):
                found = True
                break
        if found:
            true_positives += 1
        else:
            false_negatives += 1

    false_positives = 0
    if len(frame2) > len(frame1):
        false_positives = len(frame2) - len(frame1)

    precision = true_positives / max((true_positives + false_positives), 1)
    recall = true_positives / max((true_positives + false_negatives), 1)
    f1 = 2 * precision * recall / max((precision + recall), 1)
    return f1


def main():
    parser = argparse.ArgumentParser(description='Compare two video files')
    parser.add_argument('source_video_file')
    parser.add_argument('decompressed_video_file')
    parser.add_argument('-c', '-classes', default='')
    args = parser.parse_args()
    source_video_1 = args.source_video_file
    source_video_2 = args.decompressed_video_file
    target_classes = args.c.split(',')
    if target_classes == ['']:
        target_classes = []

    output_dir = os.path.abspath('../output')
    if not os.path.exists(output_dir):
        print(f'Create output directory on {output_dir}')
        os.mkdir(output_dir)
    detector = ObjectDetector()
    VideoParser.parse(source_video_1, output_dir)
    source_video_1_frames = utils.get_video_frames(source_video_1, output_dir)
    detected_objects_1 = detector.detect_all(source_video_1_frames, target_classes)
    detected_objects_1 = list([x[0] for x in detected_objects_1])

    VideoParser.parse(source_video_2, output_dir)
    source_video_2_frames = utils.get_video_frames(source_video_2, output_dir)

    # Искусственное наложение шумов
    noise_generator = NoiseGenerator(amount=0.025, var=0.01, mean=0.0, lam=0.01)
    for frame in source_video_2_frames:
        print(f'add nose to frame {frame}')
        noise_generator.add_noise(frame)

    detected_objects_2 = detector.detect_all(source_video_2_frames, target_classes)
    detected_objects_2 = list([x[0] for x in detected_objects_2])

    f1_scores = []
    for frame_index in range(min(len(detected_objects_1), len(detected_objects_2))):
        f1 = frame_f1(detected_objects_1[frame_index], detected_objects_2[frame_index])
        if f1 is not None:
            f1_scores.append(f1)
            print(f'Local F1={f1}, frame={frame_index}')
    print(f'Total F1-score: {utils.mean(f1_scores)}')


if __name__ == '__main__':
    main()
