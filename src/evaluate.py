
import os
from torchvision import transforms

from configs import eval_config as cfg
from data.dataset import video_folder
from models.conv_net import simpleConvNet
from utils.mil_utils import mil_model_wrapper
from utils.train_utils import load_model
from utils.eval_utils import print_classification_metrics
from utils.file_utils import create_class_structure
from utils.img_video_utils import save_cropped_images
from utils.img_video_utils import save_frames_from_video_folder_mil

#! TODO: write assumptions about expecting scoring_clips and non_scoring_clips

def get_directories():

    data_scoring_dir = os.path.join(cfg.DATASET_ROOT, 'scoring_clips')
    data_nonscoring_dir = os.path.join(cfg.DATASET_ROOT, 'non_scoring_clips')
    data_dir_list = [data_scoring_dir, data_nonscoring_dir]

    create_class_structure(cfg.FRAMES_DIR)
    frames_scoring_dir = os.path.join(cfg.FRAMES_DIR, '1')
    frames_nonscoring_dir = os.path.join(cfg.FRAMES_DIR, '0')
    frames_dir_list = [frames_scoring_dir, frames_nonscoring_dir]

    create_class_structure(cfg.FINAL_DATASET_DIR)
    final_scoring_dir = os.path.join(cfg.FINAL_DATASET_DIR, '1')
    final_nonscoring_dir = os.path.join(cfg.FINAL_DATASET_DIR, '0')
    final_dir_list = [final_scoring_dir, final_nonscoring_dir]

    return data_dir_list, frames_dir_list, final_dir_list


def prepare_eval_dataset():

    data_dir_list, frames_dir_list, final_dir_list = get_directories()

    for idx in range(len(data_dir_list)):
        save_frames_from_video_folder_mil(data_dir_list[idx], frames_dir_list[idx])

    for idx in range(len(frames_dir_list)):
        videos = os.listdir(frames_dir_list[idx])

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


def evaluate():

    prepare_eval_dataset()

    transform = transforms.Compose([
        transforms.Resize((cfg.RESIZE, cfg.RESIZE)),
        transforms.ToTensor(),
    ])

    dataset = video_folder(cfg.FINAL_DATASET_DIR, transform, cfg.MAX_VIDEO_FRAMES)


    # load model
    model = simpleConvNet(train_loader)
    model = model.to(cfg.DEVICE)
    model = mil_model_wrapper(model)

    # load saved weights if needed
    if cfg.LOAD_MODEL:
        load_model(model, cfg.LOAD_MODEL_PTH, cfg.DEVICE)

    print_classification_metrics(model, dataset_path, transform, max_video_frames, batch_size, device, threshold)