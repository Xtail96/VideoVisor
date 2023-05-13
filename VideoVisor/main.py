from video_parser.video_parser import VideoParser
import argparse
import os
import utils
from object_detector.object_detector import ObjectDetector


def main():
    parser = argparse.ArgumentParser(description='Compare two video files')
    parser.add_argument('first_video_file')
    parser.add_argument('second_video_file')
    args = parser.parse_args()
    source_video_1 = args.first_video_file
    source_video_2 = args.second_video_file
    output_dir = os.path.abspath('../output')
    if not os.path.exists(output_dir):
        print(f'Create output directory on {output_dir}')
        os.mkdir(output_dir)
    detector = ObjectDetector()
    VideoParser.parse(source_video_1, output_dir)
    source_video_1_frames = utils.get_video_frames(source_video_1, output_dir)
    detector.detect_all(source_video_1_frames, ['car', 'truck'])

    VideoParser.parse(source_video_2, output_dir)
    source_video_2_frames = utils.get_video_frames(source_video_2, output_dir)
    detector.detect_all(source_video_2_frames, ['cat'])


if __name__ == '__main__':
    main()
