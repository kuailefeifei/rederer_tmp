import os
import argparse
import random

from tqdm import tqdm
from PIL import Image
import numpy as np

from mlcandy.media.video_reader import VideoReader


def select_images_from_video(video_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    max_num = 20
    min_dist = 10
    count = 0
    input_video = VideoReader(video_path)
    last_idx = None
    for i, frame in enumerate(tqdm(input_video, desc="select frames from video")):
        if random.random() < 0.04:
            if last_idx is not None and (i - last_idx) < min_dist:
                continue
            last_idx = i
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
        if dir_name.startswith('.'):
            continue
        if dir_name.endswith('.mov'):
            video_path = os.path.join(input_path, dir_name)
            video_list.append(video_path)
            output_name = dir_name.split('.')[0]
            output_list.append(os.path.join(output_path, output_name))

    for video_path, image_path in zip(video_list, output_list):
        select_images_from_video(video_path, image_path)


def select_ffhq(input_path, output_path, max_num):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    count = 0
    for image_name in tqdm(os.listdir(input_path)):
        if random.random() < 0.1:
            image_path = os.path.join(input_path, image_name)
            image = Image.open(image_path).convert('RGB')
            image_save_path = os.path.join(output_path, image_name)
            image.save(image_save_path)
            count += 1
        if count >= max_num:
            break

    print('select %d images' % count)


def main(args):
    if args.select_digitman:
        batch_select_images(args.input_path, args.output_path)

    elif args.select_ffhq:
        select_ffhq(args.ffhq_path, args.selected_ffhq_path, args.max_num)


def get_parser():
    parser = argparse.ArgumentParser()

    # select images from digitman videos
    parser.add_argument('-select_digitman', '--select_digitman', default=False, action='store_true')
    parser.add_argument('-input_path', '--input_path',
						default='/root/lib/rederer_tmp/data/metahuman_2021_11_09_video', type=str)
    parser.add_argument('-output_path', '--output_path',
						default='/root/lib/rederer_tmp/data/metahuman_2021_11_09', type=str)

    # select images from ffhq dataset
    parser.add_argument('-select_ffhq', '--select_ffhq', default=False, action='store_true')
    parser.add_argument('-max_num', '--max_num', default=5000, type=int)
    parser.add_argument('-ffhq_path', '--ffhq_path', default='/root/lib/rederer_tmp/data/ffhq_arcface_aligned', type=str)
    parser.add_argument('-selected_ffhq_path', '--selected_ffhq_path', default='/root/lib/rederer_tmp/data/ffhq_aligned',
                        type=str)

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
