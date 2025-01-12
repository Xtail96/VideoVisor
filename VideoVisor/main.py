from video_parser.videoparser import VideoParser
import argparse
import os
import utils
from object_detector.object_detector import ObjectDetector
from cluster_detector.kmeans_detector import KMeansDetector
from cluster_detector.dbscan_detector import DBSCANDetector
from typing import List
from noise_generator.noise_genertor import NoiseGenerator, QPSKModulator
import cv2
import numpy as np
from yolo11_detector.yolo11_detector import YOLO11Detector
from ms_ssim_calculator import MS_SSIM_Calculator


def getPSNR(I1, I2):
    s1 = cv2.absdiff(I1, I2)  # |I1 - I2|
    s1 = np.float32(s1)  # cannot make a square on 8 bits
    s1 = s1 * s1  # |I1 - I2|^2
    sse = s1.sum()  # sum elements per channel
    if sse <= 1e-10:  # sum channels
        return 0  # for small values return zero
    else:
        shape = I1.shape
    mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
    psnr = 10.0 * np.log10((255 * 255) / mse)
    return psnr


def getMSSISM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS

    I1 = np.float32(i1)  # cannot calculate on one byte large values
    I2 = np.float32(i2)

    I2_2 = I2 * I2  # I2^2
    I1_2 = I1 * I1  # I1^2
    I1_I2 = I1 * I2  # I1 * I2
    # END INITS

    # PRELIMINARY COMPUTING
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)

    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2

    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2

    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2  # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2  # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    ssim_map = cv2.divide(t3, t1)  # ssim_map = t3./t1;

    mssim = cv2.mean(ssim_map)  # mssim = average of ssim map
    return mssim


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


def detect(detector, frames, target_classes='', silent=True):
    if target_classes == ['']:
        target_classes = []

    detected_objects_1 = detector.detect_all(frames, target_classes, silent)
    detected_objects_1 = list([x[0] for x in detected_objects_1])
    return detected_objects_1


def frame_with_objects_count(detected_objects):
    frames_with_objects = 0
    for x in detected_objects:
        if len(x) > 0:
            frames_with_objects += 1
    return frames_with_objects


def psnr_calculation(frames_1, frames_2):
    print('Starting PSNR calculation')
    psnr_scores = []
    for frame_index in range(min(len(frames_1), len(frames_2))):
        psnr = getPSNR(cv2.imread(frames_1[frame_index]), cv2.imread(frames_2[frame_index]))
        # print(f'Local PSNR={psnr}, frame={frame_index}')
        psnr_scores.append(psnr)
    print('PSNR calculation finished')
    total_psnr = utils.mean(psnr_scores)
    inverse_total_psnr = 1.0 if int(total_psnr) == 0 else 1 / total_psnr
    return total_psnr, inverse_total_psnr


def ms_ssim_calculation(frames1, frames2):
    print('Starting MS-SSIM calculation')
    calculator = MS_SSIM_Calculator()
    ms_ssim_scores = []
    for frame_index in range(min(len(frames1), len(frames2))):
        ms_ssim = calculator.calculate_ms_ssim(frames1[frame_index], frames2[frame_index])
        #ms_ssim = getMSSISM(cv2.imread(frames1[frame_index]), cv2.imread(frames2[frame_index]))
        #ms_ssim = ms_ssim[0]
        # print(f'Local MSSIM={mssim}, frame={frame_index}')
        ms_ssim_scores.append(ms_ssim)
    print('MS-SSIM calculation finished')
    total_ms_ssim = utils.mean(ms_ssim_scores)
    inverse_total_ms_ssim = 1.0 if int(total_ms_ssim * 100) == 0 else 1 / total_ms_ssim
    return total_ms_ssim, inverse_total_ms_ssim


def add_noise(frames, use_modulation=False):
    print('Starting add noise')
    noise_generator = NoiseGenerator(amount=0.01, var=0.01, mean=0.0, lam=0.01)
    for frame in frames:
        # print(f'add noise to frame {frame}')
        noise_generator.add_noise(frame, use_modulation)
    print('Add noise finishing')


def f1_calculation(frames1, frames2, detector, target_classes=''):
    print('Starting F1-calculation')

    # Detect objects
    detected_objects_1 = detect(detector, frames1, target_classes)
    print(f'Detected {frame_with_objects_count(detected_objects_1)} objects on first video')

    detected_objects_2 = detect(detector, frames2, target_classes)
    print(f'Detected {frame_with_objects_count(detected_objects_2)} objects on second video')

    f1_scores = []
    for frame_index in range(min(len(detected_objects_1), len(detected_objects_2))):
        f1 = frame_f1(detected_objects_1[frame_index], detected_objects_2[frame_index])
        if f1 is not None:
            f1_scores.append(f1)
            # print(f'Local F1={f1}, frame={frame_index}')
    print('F1 calculation finished')
    return utils.mean(f1_scores)

def main():
    parser = argparse.ArgumentParser(description='Compare two video files')
    parser.add_argument('source_video_file')
    parser.add_argument('decompressed_video_file')
    parser.add_argument('-c', '-classes', default='car,truck,bus')
    args = parser.parse_args()
    source_video_1 = args.source_video_file
    source_video_2 = args.decompressed_video_file
    target_classes = args.c.split(',')

    output_dir = os.path.abspath('../examples/output')
    if not os.path.exists(output_dir):
        print(f'Create output directory on {output_dir}')
        os.mkdir(output_dir)

    # Parse source video
    VideoParser.parse(source_video_1, output_dir)
    source_video_1_frames = utils.get_video_frames(source_video_1, output_dir)

    # Parse transferred video
    VideoParser.parse(source_video_2, output_dir)
    source_video_2_frames = utils.get_video_frames(source_video_2, output_dir)
    add_noise(source_video_2_frames)

    # Calculate metrics
    total_psnr, inverse_total_psnr = psnr_calculation(source_video_1_frames, source_video_2_frames)
    total_ms_ssim, inverse_total_ms_ssim = ms_ssim_calculation(source_video_1_frames, source_video_2_frames)

    # Create detectors
    kmeans_detector = KMeansDetector()
    y4_detector = ObjectDetector()
    y11_detector = YOLO11Detector()

    detector = y11_detector
    total_f1 = f1_calculation(source_video_1_frames, source_video_2_frames, detector, target_classes)

    # Print metrics
    print(f'Total F1-score: {total_f1}')
    print(f'Total PSNR-score: {total_psnr}. Inverse: {inverse_total_psnr}')
    print(f'Total MSSIM-score: {total_ms_ssim}. Inverse: {inverse_total_ms_ssim}')


if __name__ == '__main__':
    main()
