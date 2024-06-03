import os
import shutil
from video_parser.video2images import Video2Images


class VideoParser:
    @staticmethod
    def parse(video_file_path: str, output_dir: str):
        if not os.path.exists(video_file_path):
            raise Exception(f'Source file not found {video_file_path}')
        filename = os.path.basename(video_file_path)
        result_folder = os.path.join(output_dir, filename)
        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)
        Video2Images(video_filepath=video_file_path, out_dir=output_dir)
