'''
This program is written to select images with natural expression from
digitman dataset and cfd dataset
'''

import os
import argparse

from PIL import Image
from tqdm import tqdm


def generate_naive_txt(image_dir, text_file):
    file_list = []
    for file_name in sorted(os.listdir(image_dir)):
        if file_name.startswith('.'):
            continue
        file_list.append(file_name + '\n')
    file_list[-1] = file_list[-1].replace('\n', '')
    with open(text_file, 'w') as f:
        f.writelines(file_list)


def select_images(text_file, source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(text_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            folder_name, image_name = line.strip().split(',')
            source_image_path = os.path.join(source_dir, folder_name, image_name)
            target_folder_path = os.path.join(target_dir, folder_name)
            if not os.path.exists(target_folder_path):
                os.makedirs(target_folder_path)
            target_image_path = os.path.join(target_folder_path, image_name)
            image = Image.open(source_image_path).convert('RGB')
            image.save(target_image_path)


def main(args):
    if args.generate_text:
        generate_naive_txt(args.digit_image_dir, args.digit_text_file)

    elif args.select_images:
        select_images(args.text_file, args.source_dir, args.target_dir)


def get_args():
    parser = argparse.ArgumentParser()
    # generate naive text file
    parser.add_argument('-generate_text', '--generate_text', default=False, action='store_true')
    parser.add_argument('-digit_image_dir', '--digit_image_dir', default='/Users/mac/Desktop/digitman_front_aligned', type=str)
    parser.add_argument('-digit_text_file', '--digit_text_file', default='/Users/mac/Documents/GitHub/rederer_tmp/data_utils/digitman_mini.txt', type=str)

    # select images according to the given text file
    parser.add_argument('-select_images', '--select_images', default=False, action='store_true')
    parser.add_argument('-text_file', '--text_file', default='/Users/mac/Documents/GitHub/rederer_tmp/data_utils/digitman_mini.txt', type=str)
    parser.add_argument('-source_dir', '--source_dir', default='/Users/mac/Desktop/digitman_front_aligned', type=str)
    parser.add_argument('-target_dir', '--target_dir', default='/Users/mac/Desktop/digitman_front_aligned_mini', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
