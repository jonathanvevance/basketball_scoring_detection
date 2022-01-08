"""Python file to evaluate a trained model."""

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from configs import eval_config as cfg
from data.dataset import eval_folder
from models.conv_net import simpleConvNet
from utils.mil_utils import mil_model_wrapper
from utils.train_utils import load_model
from utils.eval_utils import print_classification_metrics
from utils.file_utils import listdir
from utils.file_utils import clear_subfolders_in_folder
from utils.file_utils import create_class_structure
from utils.img_video_utils import filter_videos_func
from utils.img_video_utils import save_cropped_images
from utils.img_video_utils import save_frames_from_video_folder


def get_directories():
    """Return (create if required) directory paths."""

    data_scoring_dir = os.path.join(cfg.DATASET_ROOT, 'scoring_clips')
    data_nonscoring_dir = os.path.join(cfg.DATASET_ROOT, 'non_scoring_clips')
    data_dir_list = [data_scoring_dir, data_nonscoring_dir]

    create_class_structure(cfg.FRAMES_DIR) # create 0/1 folders
    frames_scoring_dir = os.path.join(cfg.FRAMES_DIR, '1')
    frames_nonscoring_dir = os.path.join(cfg.FRAMES_DIR, '0')
    frames_dir_list = [frames_scoring_dir, frames_nonscoring_dir]

    create_class_structure(cfg.FINAL_DATASET_DIR) # create 0/1 folders
    final_scoring_dir = os.path.join(cfg.FINAL_DATASET_DIR, '1')
    final_nonscoring_dir = os.path.join(cfg.FINAL_DATASET_DIR, '0')
    final_dir_list = [final_scoring_dir, final_nonscoring_dir]

    return data_dir_list, frames_dir_list, final_dir_list


def prepare_eval_dataset():
    """Prepare dataset for classification model evaluation."""

    data_dir_list, frames_dir_list, final_dir_list = get_directories()

    for idx in range(len(data_dir_list)):
        save_frames_from_video_folder(data_dir_list[idx], frames_dir_list[idx])

    # runn yolov3 model for basket detection
    for idx in range(len(frames_dir_list)):
        videos = list(filter(filter_videos_func, listdir(frames_dir_list[idx])))

        for video in videos:
            video_frames_dir = os.path.join(frames_dir_list[idx], video)

            commands = [
                'cd ./src/yolov3_helper',
                'sudo bash predict.sh' + ' ../../../' + video_frames_dir + ' ../../../' + video_frames_dir
            ]
            os.system(';'.join(commands)) # this stores json in video_frames_dir itself

            # Step 3: Crop according to bounding boxes and save them
            final_frames_dir = os.path.join(final_dir_list[idx], video)
            save_cropped_images(video_frames_dir, final_frames_dir, standardise = True)

    return data_dir_list, frames_dir_list, final_dir_list


def evaluate():
    """Evaluate function."""

    __, frames_dir_list, final_dir_list = prepare_eval_dataset()

    print()
    transform = transforms.Compose([
        transforms.Resize((cfg.RESIZE, cfg.RESIZE)),
        transforms.ToTensor(),
    ])

    dataset = eval_folder(cfg.FINAL_DATASET_DIR, transform, cfg.MAX_VIDEO_FRAMES)
    dataloader = DataLoader(dataset = dataset, batch_size = cfg.BATCH_SIZE, shuffle = True)

    # set device
    cfg.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = simpleConvNet(dataloader)
    model = model.to(cfg.DEVICE)
    model = mil_model_wrapper(model)

    # load saved weights
    load_model(model, cfg.LOAD_MODEL_PTH, cfg.DEVICE)

    print_classification_metrics(
        model,
        cfg.FINAL_DATASET_DIR,
        transform,
        cfg.MAX_VIDEO_FRAMES,
        cfg.BATCH_SIZE,
        cfg.DEVICE,
        cfg.SAVE_RESULTS_DIR,
        cfg.THRESHOLD,
    )

    # clear temporary files stored
    for frames_dir in frames_dir_list:
        clear_subfolders_in_folder(frames_dir)

    for final_dir in final_dir_list:
        clear_subfolders_in_folder(final_dir)


evaluate()