import os
import argparse
import random

import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np

from mlcandy.media.video_reader import VideoReader


def select_images_from_video(video_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    max_num = 20
    count = 0
    input_video = VideoReader(video_path)
    for i, frame in enumerate(tqdm(input_video, desc="select frames from video")):
        if random.random() < 0.02:
            image = Image.fromarray(frame).convert('RGB')
            image_name = '%04d.jpg' % i
            image_save_path = os.path.join(output_path, image_name)
            image.save(image_save_path)
            count += 1
        if count >= max_num:
            break
    print('find %d images!' % count)
    input_video.close()


def batch_select_images(input_path, output_path):
    video_list, output_list = [], []
    for dir_name in sorted(os.listdir(input_path)):
        if dir_name.endswith('.mov'):
            video_path = os.path.join(input_path, dir_name)
            video_list.append(video_path)
            output_name = dir_name.split('.')[0]
            output_list.append(os.path.join(output_path, output_name))

    print(video_list)

    # for video_path, image_path in zip(video_list, output_list):
    #     select_images_from_video(video_path, image_path)


def main(args):
    batch_select_images(args.input_path, args.output_path)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-input_path', '--input_path',
						default='/root/lib/rederer_tmp/data/metahuman', type=str)
    parser.add_argument('-output_path', '--output_path',
						default='/root/lib/rederer_tmp/data/digitman', type=str)

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
