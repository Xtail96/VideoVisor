from video2images import Video2Images


def main():
    source_video_file = "../examples/input/lada.mp4"
    output_dir = "../output"
    Video2Images(video_filepath=source_video_file, out_dir=output_dir)


if __name__ == "__main__":
    main()
