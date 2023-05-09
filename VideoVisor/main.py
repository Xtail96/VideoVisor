from video_parser.video_parser import VideoParser
import argparse


def main():
    parser = argparse.ArgumentParser(description='Compare two video files')
    parser.add_argument('first_video_file')
    parser.add_argument('second_video_file')
    args = parser.parse_args()
    source_video_1 = args.first_video_file
    source_video_2 = args.second_video_file
    output_dir = '../output'
    VideoParser.parse(source_video_1, output_dir)
    VideoParser.parse(source_video_2, output_dir)


if __name__ == '__main__':
    main()
