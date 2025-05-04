import os
from typing import List
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from tqdm import tqdm


nChannel = 100
nConv = 2
scribble = False
minLabels = 3
lr = 0.1
maxIter = 100 #1000
visualize = 0
stepsize_sim = 1
stepsize_scr = 0.5
stepsize_con = 1
max_iter_not_change = 5


# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append(nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(nChannel))
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(nConv-1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class CNNBasedDetector:
    def __init__(self):
        pass

    def create_segmented_image(self, classes_coords):
        pass

    @staticmethod
    def get_cluster_coords(segmented_image, class_index):
        coords = []
        x, y = segmented_image.shape
        for i in range(x):
            for j in range(y):
                if segmented_image[i, j] == class_index:
                    coords.append(utils.Point2D(j, i))
        return coords

    @staticmethod
    def create_cluster(coords, frame, label) -> utils.Cluster2D:
        return utils.Cluster2D(coords, label)

    @staticmethod
    def get_cluster_center(coords):
        a = np.array(coords)
        mean = np.mean(a, axis=0)
        return mean[0], mean[1]

    @staticmethod
    def create_cluster_bbox(cluster_coords, frame, label) -> utils.DetectedObject:
        #cluster_coords = sorted(cluster_coords, key=lambda k: [k[0], k[1]])
        cluster_x = [k[0] for k in cluster_coords]
        cluster_y = [k[1] for k in cluster_coords]

        top_left = (min(cluster_x), min(cluster_y))
        right_bottom = (max(cluster_x), max(cluster_y))
        width = right_bottom[0] - top_left[0]
        height = right_bottom[1] - top_left[1]

        # Вычисление центра масс кластера и создание окрестности
        centroid_x, centroid_y = CNNBasedDetector.get_cluster_center(cluster_coords)
        bbox_top_left_x = int(centroid_x - width / 10)
        bbox_top_left_y = int(centroid_y - height / 10)
        bbox_width = int(width / 10)
        bbox_height = int(height / 10)

        if width < 0 or height < 0:
            raise Exception('width and height can not be < 0')

        return utils.DetectedObject(label, [bbox_top_left_x, bbox_top_left_y, bbox_width, bbox_height], frame)

    def detect(self, img_path: str, debug=False) -> (List[utils.DetectedObject], str):
        # load image
        im = cv2.imread(img_path)
        data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
        data = Variable(data)

        # load scribble
        if scribble:
            mask = cv2.imread(img_path.replace('.' + img_path.split('.')[-1], '_scribble.png'), -1)
            mask = mask.reshape(-1)
            mask_inds = np.unique(mask)
            mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
            inds_sim = torch.from_numpy(np.where(mask == 255)[0])
            inds_scr = torch.from_numpy(np.where(mask != 255)[0])
            target_scr = torch.from_numpy(mask.astype(np.int))
            target_scr = Variable(target_scr)
            # set minLabels
            # minLabels = len(mask_inds)

        # train
        model = MyNet(data.size(1))
        model.train()

        # similarity loss definition
        loss_fn = torch.nn.CrossEntropyLoss()

        # scribble loss definition
        loss_fn_scr = torch.nn.CrossEntropyLoss()

        # continuity loss definition
        loss_hpy = torch.nn.L1Loss(size_average=True)
        loss_hpz = torch.nn.L1Loss(size_average=True)

        HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], nChannel)
        HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, nChannel)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        label_colours = np.random.randint(255, size=(100, 3))

        n_labels_prev = 0
        not_change = 0
        for batch_idx in range(maxIter):
            # forwarding
            optimizer.zero_grad()
            output = model(data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)

            outputHP = output.reshape((im.shape[0], im.shape[1], nChannel))
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)

            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))

            # loss
            if scribble:
                loss = (stepsize_sim * loss_fn(output[inds_sim], target[inds_sim]) +
                        stepsize_scr * loss_fn_scr(output[inds_scr], target_scr[inds_scr]) + stepsize_con * (lhpy + lhpz))
            else:
                loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)

            loss.backward()
            optimizer.step()

            print(batch_idx, '/', maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

            if nLabels <= minLabels:
                print("nLabels", nLabels, "reached minLabels", minLabels, ".")
                break

            if not_change >= max_iter_not_change:
                print(f'nLabels {nLabels} not change for {max_iter_not_change} iterations. stop')
                break

            if nLabels == n_labels_prev:
                not_change = not_change + 1
            else:
                not_change = 0
            n_labels_prev = nLabels

        # save output image
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[c % nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        cv2.imwrite(f'{img_path}_cnn.png', im_target_rgb)

        detected_clusters = []
        classes_count = im_target.max() - 1
        frame_number = int(os.path.basename(img_path).split('.')[0])
        for i in range(classes_count):
            img = im_target.reshape((im.shape[0], im.shape[1]))
            coords = self.get_cluster_coords(img, i + 1)
            if len(coords) > 0:
                detected_clusters.append(self.create_cluster(coords, frame_number, f'{i + 1}'))

        if debug:
            for cluster in detected_clusters:
                cluster.draw(img_path, list(np.random.choice(range(256), size=3)))
        return detected_clusters, img_path

    def detect_all(self, images: List[str], target_classes: List[str], silent: bool) -> List[utils.DetectedObject]:
        return list(self.detect(img, True) for img in tqdm(images[::25], f'{type(self).__name__}'))
