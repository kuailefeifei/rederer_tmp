import os
import argparse
import json

from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from facehack.FaceShifter.data_utils.align_retinaface import align_with_retinaface
from mlcandy.face_detection.retinaface_detector import RetinaFaceDetector


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
retinaface_detector = RetinaFaceDetector('/root/lib/facehack/FaceShifter/model/Resnet50_Final.pth', DEVICE)


def prepare_cfd(dataset_path):
    image_list = []
    for dir_name in os.listdir(dataset_path):
        if dir_name.startswith('.'):
            continue
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for image_name in os.listdir(dir_path):
            if image_name.startswith('.'):
                continue
            image_path = os.path.join(dir_path, image_name)
            image_list.append(image_path)
    print('there are %d images in total.' % len(image_list))
    return image_list


def prepare_digitman(dataset_path):
    image_list = []
    for dir_name in os.listdir(dataset_path):
        dir_path = os.path.join(dataset_path, dir_name)
        for image_name in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image_name)
            image_list.append(image_path)
    print('there are %d images in total.' % len(image_list))
    return image_list


def align_image(input_path, detect_size=1024):
    try:
        image = Image.open(input_path)
        image, inv_tfm = align_with_retinaface(image, retinaface_detector, detect_size=detect_size)
        assert image is not None and inv_tfm is not None
        return image
    except Exception as e:
        return None


def visualize_image(image_list, log_interval=10, num_show=30):
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    writer = SummaryWriter('writer')
    tensor_list = []
    for idx, image_path in tqdm(enumerate(image_list)):
        image = Image.open(image_path)
        # image = image.resize((256, 256))
        img_tensor = trans(image)
        if idx % log_interval == 0:
            tensor_list.append(img_tensor)
        if len(tensor_list) == num_show:
            grid = make_grid(tensor_list, nrow=3)
            writer.add_image('image_samples', grid, idx)
            tensor_list = []

    if len(tensor_list) > 0:
        grid = make_grid(tensor_list, nrow=3)
        writer.add_image('image_samples', grid, idx)


def main(args):
    if args.align_cfd:
        if not os.path.exists(args.aligned_cfd_path):
            os.makedirs(args.aligned_cfd_path)

        image_list = prepare_cfd(args.cfd_path)
        for image_path in tqdm(image_list):
            image_aligned = align_image(image_path)
            if image_aligned is not None:
                dir_name, image_name = image_path.split('/')[-2], image_path.split('/')[-1]
                dir_path = os.path.join(args.aligned_cfd_path, dir_name)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                image_save_path = os.path.join(dir_path, image_name)
                image_aligned.save(image_save_path)

    elif args.align_digitman:
        if not os.path.exists(args.aligned_digitman_path):
            os.makedirs(args.aligned_digitman_path)

        image_list = prepare_digitman(args.digitman_path)
        for image_path in tqdm(image_list):
            image_aligned = align_image(image_path)
            if image_aligned is not None:
                dir_name, image_name = image_path.split('/')[-2], image_path.split('/')[-1]
                dir_path = os.path.join(args.aligned_digitman_path, dir_name)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                image_save_path = os.path.join(dir_path, image_name)
                image_aligned.save(image_save_path)

    elif args.visualize:
        image_list = prepare_digitman(args.visualize_path)
        visualize_image(image_list)


def get_args():
    parser = argparse.ArgumentParser()

    # align cfd images
    parser.add_argument('-align_cfd', '--align_cfd', default=False, action='store_true')
    parser.add_argument('-cfd_path', '--cfd_path', default='/root/lib/rederer_tmp/data/cfd_version_3_0/Images/CFD', type=str)
    parser.add_argument('-aligned_cfd_path', '--aligned_cfd_path', default='/root/lib/rederer_tmp/data/cfd_aligned', type=str)

    # align digit images
    parser.add_argument('-align_digitman', '--align_digitman', default=False, action='store_true')
    parser.add_argument('-digitman_path', '--digitman_path', default='/root/lib/rederer_tmp/data/digitman',
                        type=str)
    parser.add_argument('-aligned_digitman_path', '--aligned_digitman_path', default='/root/lib/rederer_tmp/data/digitman_aligned',
                        type=str)

    # visualize images
    parser.add_argument('-visualize', '--visualize', default=False, action='store_true')
    parser.add_argument('-visualize_path', '--visualize_path', default='/root/lib/rederer_tmp/data/digitman_aligned', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
