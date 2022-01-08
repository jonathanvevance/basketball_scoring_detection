# -*- coding: utf-8 -*-
"""Python file to prepare dataset for training."""

import os
import click
from tqdm import tqdm

import random
random.seed(0)

from utils.file_utils import listdir
from utils.file_utils import copy_tree
from utils.file_utils import create_dataset_structure
from utils.img_video_utils import save_cropped_images
from utils.img_video_utils import save_frames_from_video_folder

FRAMES_DIR = 'data/training/frames'
DATASET_ROOT = 'data/training/hackolympics_data'
CROPPED_DATASET_DIR = 'data/training/cropped'
FINAL_DATASET_DIR = 'data/training/final'


def get_directories():
    """Return (create if required) directory paths."""

    train_scoring_dir = os.path.join(DATASET_ROOT, 'Training_Data/scoring_clips')
    train_nonscoring_dir = os.path.join(DATASET_ROOT, 'Training_Data/non_scoring_clips')
    test_scoring_dir = os.path.join(DATASET_ROOT, 'Public_Test_Data/scoring_clips')
    test_nonscoring_dir = os.path.join(DATASET_ROOT, 'Public_Test_Data/non_scoring_clips')

    data_dir_list = [
        train_scoring_dir,
        train_nonscoring_dir,
        test_scoring_dir,
        test_nonscoring_dir
    ]

    create_dataset_structure(FRAMES_DIR)
    train_frames_scoring_dir = os.path.join(FRAMES_DIR, 'train/1')
    train_frames_nonscoring_dir = os.path.join(FRAMES_DIR, 'train/0')
    test_frames_scoring_dir = os.path.join(FRAMES_DIR, 'test/1')
    test_frames_nonscoring_dir = os.path.join(FRAMES_DIR, 'test/0')

    frames_dir_list = [
        train_frames_scoring_dir,
        train_frames_nonscoring_dir,
        test_frames_scoring_dir,
        test_frames_nonscoring_dir
    ]

    create_dataset_structure(CROPPED_DATASET_DIR)
    train_cropped_scoring_dir = os.path.join(CROPPED_DATASET_DIR, 'train/1')
    train_cropped_nonscoring_dir = os.path.join(CROPPED_DATASET_DIR, 'train/0')
    test_cropped_scoring_dir = os.path.join(CROPPED_DATASET_DIR, 'test/1')
    test_cropped_nonscoring_dir = os.path.join(CROPPED_DATASET_DIR, 'test/0')

    cropped_dir_list = [
        train_cropped_scoring_dir,
        train_cropped_nonscoring_dir,
        test_cropped_scoring_dir,
        test_cropped_nonscoring_dir
    ]

    create_dataset_structure(FINAL_DATASET_DIR, val_required = True)
    train_final_scoring_dir = os.path.join(FINAL_DATASET_DIR, 'train/1')
    train_final_nonscoring_dir = os.path.join(FINAL_DATASET_DIR, 'train/0')
    val_final_scoring_dir = os.path.join(FINAL_DATASET_DIR, 'val/1')
    val_final_nonscoring_dir = os.path.join(FINAL_DATASET_DIR, 'val/0')
    test_final_scoring_dir = os.path.join(FINAL_DATASET_DIR, 'test/1')
    test_final_nonscoring_dir = os.path.join(FINAL_DATASET_DIR, 'test/0')

    final_dir_dict = {
        'train': {
            '1': train_final_scoring_dir,
            '0': train_final_nonscoring_dir,
        },

        'val': {
            '1': val_final_scoring_dir,
            '0': val_final_nonscoring_dir,
        },

        'test': {
            '1': test_final_scoring_dir,
            '0': test_final_nonscoring_dir,
        }
    }

    return data_dir_list, frames_dir_list, cropped_dir_list, final_dir_dict


def split_train_val_sets(final_dir_dict, val_ratio):
    """Split train set into train-val sets."""

    train_cropped_scoring_folder = os.path.join(CROPPED_DATASET_DIR, 'train/1')
    train_cropped_nonscoring_folder = os.path.join(CROPPED_DATASET_DIR, 'train/0')
    test_cropped_scoring_folder = os.path.join(CROPPED_DATASET_DIR, 'test/1')
    test_cropped_nonscoring_folder = os.path.join(CROPPED_DATASET_DIR, 'test/0')

    train_scoring_folder = final_dir_dict['train']['1']
    train_nonscoring_folder = final_dir_dict['train']['0']
    val_scoring_folder = final_dir_dict['val']['1']
    val_nonscoring_folder = final_dir_dict['val']['0']
    test_scoring_folder = final_dir_dict['test']['1']
    test_nonscoring_folder = final_dir_dict['test']['0']

    # move scoring (cropped) frames folders
    train_cropped_scoring_videos = listdir(train_cropped_scoring_folder)
    random.shuffle(train_cropped_scoring_videos)
    num_val = int(len(train_cropped_scoring_videos) * val_ratio)

    for idx in tqdm(range(num_val + 1)):
        source_video_dir = os.path.join(train_cropped_scoring_folder, train_cropped_scoring_videos[idx])
        destination_video_dir = os.path.join(val_scoring_folder, train_cropped_scoring_videos[idx])
        copy_tree(source_video_dir, destination_video_dir) # copy folder

    for idx in tqdm(range(num_val + 1, len(train_cropped_scoring_videos))):
        source_video_dir = os.path.join(train_cropped_scoring_folder, train_cropped_scoring_videos[idx])
        destination_video_dir = os.path.join(train_scoring_folder, train_cropped_scoring_videos[idx])
        copy_tree(source_video_dir, destination_video_dir) # copy folder

    # move nonscoring (cropped) frames folders
    train_cropped_nonscoring_videos = listdir(train_cropped_nonscoring_folder)
    random.shuffle(train_cropped_nonscoring_videos)
    num_val = int(len(train_cropped_nonscoring_videos) * val_ratio)

    for idx in tqdm(range(num_val + 1)):
        source_video_dir = os.path.join(train_cropped_nonscoring_folder, train_cropped_nonscoring_videos[idx])
        destination_video_dir = os.path.join(val_nonscoring_folder, train_cropped_nonscoring_videos[idx])
        copy_tree(source_video_dir, destination_video_dir) # copy folder

    for idx in tqdm(range(num_val + 1, len(train_cropped_nonscoring_videos))):
        source_video_dir = os.path.join(train_cropped_nonscoring_folder, train_cropped_nonscoring_videos[idx])
        destination_video_dir = os.path.join(train_nonscoring_folder, train_cropped_nonscoring_videos[idx])
        copy_tree(source_video_dir, destination_video_dir) # copy folder

    # move test frames folders
    test_cropped_scoring_videos = listdir(test_cropped_scoring_folder)
    for video in tqdm(test_cropped_scoring_videos):
        source_video_dir = os.path.join(test_cropped_scoring_folder, video)
        destination_video_dir = os.path.join(test_scoring_folder, video)
        copy_tree(source_video_dir, destination_video_dir)

    test_cropped_nonscoring_videos = listdir(test_cropped_nonscoring_folder)
    for video in tqdm(test_cropped_nonscoring_videos):
        source_video_dir = os.path.join(test_cropped_nonscoring_folder, video)
        destination_video_dir = os.path.join(test_nonscoring_folder, video)
        copy_tree(source_video_dir, destination_video_dir)


@click.command()
@click.option('--videos', is_flag = True)
@click.option('--yolov3', is_flag = True)
@click.option('--split', is_flag = True)
@click.option('--val-ratio', type = click.FloatRange(0.0, 1.0), default = 0.3)
def main(videos, yolov3, split, val_ratio):
    """
    Main function. The flags (videos, yolov3, split) are added so that make_dataset can be done in parts.
    The entire make_dataset process may take a long time (yolov3 detector being the slowest step).
    """

    data_dir_list, frames_dir_list, cropped_dir_list, final_dir_dict = get_directories()

    if videos:
        # Step 1: collect frames from videos and put them in frames folders
        for idx in range(len(data_dir_list)):
            save_frames_from_video_folder(data_dir_list[idx], frames_dir_list[idx])

    if yolov3:

        # Step 2: Run yolov3 basket detector on frames and get bounding box coords
        for idx in range(len(frames_dir_list)):
            videos = listdir(frames_dir_list[idx])

            for video in videos:
                video_frames_dir = os.path.join(frames_dir_list[idx], video)

                commands = [
                    'cd ./src/yolov3_helper',
                    'sudo bash predict.sh' + ' ../../../' + video_frames_dir + ' ../../../' + video_frames_dir
                ]
                os.system(';'.join(commands)) # this stores json in video_frames_dir itself

                # Step 3: Crop according to bounding boxes and save them
                cropped_frames_dir = os.path.join(cropped_dir_list[idx], video)
                save_cropped_images(video_frames_dir, cropped_frames_dir, standardise = True)

    if split:
        # Step 4: Split train-test
        split_train_val_sets(final_dir_dict, val_ratio)


if __name__ == '__main__':
    main()
