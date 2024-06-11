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


def main():
    parser = argparse.ArgumentParser(description='Compare two video files')
    parser.add_argument('source_video_file')
    parser.add_argument('decompressed_video_file')
    parser.add_argument('-c', '-classes', default='car,truck,bus')
    args = parser.parse_args()
    source_video_1 = args.source_video_file
    source_video_2 = args.decompressed_video_file
    target_classes = args.c.split(',')
    if target_classes == ['']:
        target_classes = []

    output_dir = os.path.abspath('../examples/output')
    if not os.path.exists(output_dir):
        print(f'Create output directory on {output_dir}')
        os.mkdir(output_dir)

    VideoParser.parse(source_video_1, output_dir)
    source_video_1_frames = utils.get_video_frames(source_video_1, output_dir)
    dbscan_detector = DBSCANDetector()
    detected_objects_1 = dbscan_detector.detect_all(source_video_1_frames)
    detected_objects_2 = detected_objects_1
    return

    #kmeans_detector = KMeansDetector()
    #detected_objects_1 = kmeans_detector.detect_all(source_video_1_frames)

    #detector = ObjectDetector()
    #detected_objects_1 = detector.detect_all(source_video_1_frames, target_classes)
    #detected_objects_1 = list([x[0] for x in detected_objects_1])

    VideoParser.parse(source_video_2, output_dir)
    source_video_2_frames = utils.get_video_frames(source_video_2, output_dir)

    # Искусственное наложение шумов на кадры
    noise_generator = NoiseGenerator(amount=0.01, var=0.01, mean=0.0, lam=0.01)
    for frame in source_video_2_frames:
        print(f'add noise to frame {frame}')
        noise_generator.add_noise(frame, False)

    #detected_objects_2 = kmeans_detector.detect_all(source_video_2_frames)
    #detected_objects_2 = detector.detect_all(source_video_2_frames, target_classes)
    #detected_objects_2 = list([x[0] for x in detected_objects_2])

    psnr_scores = []
    for frame_index in range(min(len(source_video_1_frames), len(source_video_2_frames))):
        psnr = getPSNR(cv2.imread(source_video_1_frames[frame_index]), cv2.imread(source_video_2_frames[frame_index]))
        print(f'Local PSNR={psnr}, frame={frame_index}')
        psnr_scores.append(psnr)

    #mssim_scores = []
    #for frame_index in range(min(len(source_video_1_frames), len(source_video_2_frames))):
    #    mssim = getMSSISM(cv2.imread(source_video_1_frames[frame_index]), cv2.imread(source_video_2_frames[frame_index]))
    #    print(f'Local MSSIM={mssim}, frame={frame_index}')
    #    mssim_scores.append(mssim)

    f1_scores = []
    for frame_index in range(min(len(detected_objects_1), len(detected_objects_2))):
        f1 = frame_f1(detected_objects_1[frame_index], detected_objects_2[frame_index])
        if f1 is not None:
            f1_scores.append(f1)
            print(f'Local F1={f1}, frame={frame_index}')
    print(f'Total F1-score: {utils.mean(f1_scores)}')

    total_psnr = utils.mean(psnr_scores)
    inverse_total_psnr = 1.0 if int(total_psnr) == 0 else 1 / total_psnr
    print(f'Total PSNR-score: {total_psnr}. Inverse: {inverse_total_psnr}')

    #total_mssim = utils.mean(mssim_scores)
    #inverse_total_mssim = 1.0 if int(total_mssim) == 0 else 1 / total_mssim
    #print(f'Total MSSIM-score: {total_mssim}. Inverse: {inverse_total_mssim}')



if __name__ == '__main__':
    main()
