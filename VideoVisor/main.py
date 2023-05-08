from video_parser.video_parser import VideoParser


def main():
    source_video_file = "../examples/input/lada.mp4"
    VideoParser.parse(source_video_file)


if __name__ == "__main__":
    main()
