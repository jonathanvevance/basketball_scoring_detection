#!/.env python3

import os
import json
import cv2
from PIL import Image
from tqdm import tqdm

from utils.file_utils import clear_folder
from utils.file_utils import make_folder


VIDEO_UPLOAD_PATH = 'data/inference/video_upload/video.mp4'
VIDEO_UPLOAD_DIRECTORY = 'data/inference/video_upload'
FRAMES_UPLOAD_DIRECTORY = 'data/inference/frames_upload'
UPLOAD_FRAMES_COORDS_JSON = 'data/inference/frames_upload/bounding_boxes.json'

def filter_images_func(image_name):
    if len(image_name) < 4:
        return False
    if image_name[-3:] in ['jpg', 'png']:
        return True
    return False


def get_cropped_pil_images_inference(crop_size = 100, standardise = True): # only during inference

    frame_imgs = filter(filter_images_func, os.listdir(FRAMES_UPLOAD_DIRECTORY))
    frame_imgs = sorted(frame_imgs, key = lambda x: int(x[:-4]))

    with open(UPLOAD_FRAMES_COORDS_JSON) as f:
        img_to_coords = json.load(f)

    cropped_pil_images = []

    for frame_img in frame_imgs:

        if frame_img not in img_to_coords:
            cropped_pil_images.append(None)

        else:
            frame_img_path = os.path.join(FRAMES_UPLOAD_DIRECTORY, frame_img)
            x1, y1, x2, y2 = img_to_coords[frame_img]

            if standardise:
                x_mid = (x1 + x2) // 2
                y_mid = (y1 + y2) // 2
                x1, x2 = x_mid - (crop_size // 2), x_mid + (crop_size // 2)
                y1, y2 = y_mid - (crop_size // 2), y_mid + (crop_size // 2)

            pil_img = Image.open(frame_img_path)
            cropped_pil_img = pil_img.crop((x1, y1, x2, y2))
            cropped_pil_images.append(cropped_pil_img)

    return cropped_pil_images


def save_frames_from_video_inference(): # only during inference
    # https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/

    vidObj = cv2.VideoCapture(VIDEO_UPLOAD_PATH)
    count = 0

    while True:
        success, image = vidObj.read()
        if not success:
            break

        cv2.imwrite(os.path.join(FRAMES_UPLOAD_DIRECTORY, str(count) + '.jpg'), image)
        count += 1

    clear_folder(VIDEO_UPLOAD_DIRECTORY) # clear this video


# ------------------------------------ TRAINING FUNCTIONS --------------------------------------------------

def get_last_frame_id(target_folder):
    frame_imgs = list(filter(filter_images_func, os.listdir(target_folder)))
    if len(frame_imgs) > 0:
        return int(max(frame_imgs, key = lambda x: int(x[:-4]))[:-4])
    return 0


def save_frames_from_video_folder_clf(video_folder, target_folder): # only during training (dataset creation)
    """Used for classification based training."""

    count = get_last_frame_id(target_folder) + 1
    video_filepaths = [os.path.join(video_folder, filename) for filename in os.listdir(video_folder)]

    for video_filepath in tqdm(video_filepaths, desc = f'Transferring to {target_folder}'):
        vidObj = cv2.VideoCapture(video_filepath)

        while True:
            success, image = vidObj.read()
            if not success:
                break

            cv2.imwrite(os.path.join(target_folder, str(count) + '.jpg'), image)
            count += 1


def save_frames_from_video_folder_mil(video_folder, target_folder): # only during training (dataset creation)
    """Used for multi instance based training."""

    video_files = os.listdir(video_folder)

    for video_file in tqdm(video_files, desc = f'Transferring to video folders in {target_folder}'):


        video_target_folder = os.path.join(target_folder, video_file)
        make_folder(video_target_folder)

        count = 1
        video_filepath = os.path.join(video_folder, video_file)
        vidObj = cv2.VideoCapture(video_filepath)

        while True:
            success, image = vidObj.read()
            if not success:
                break

            cv2.imwrite(os.path.join(video_target_folder, str(count) + '.jpg'), image)
            count += 1


def save_cropped_images(frames_folder, target_folder, crop_size = 100, standardise = False): # only during training (dataset creation)

    make_folder(target_folder) # make target folder

    frame_imgs = filter(filter_images_func, os.listdir(frames_folder))
    frame_imgs = sorted(frame_imgs, key = lambda x: int(x[:-4]))
    frames_coords_json_path = os.path.join(frames_folder, 'bounding_boxes.json')

    with open(frames_coords_json_path) as f:
        img_to_coords = json.load(f)

    for frame_img in tqdm(frame_imgs, desc = f'Transferring to {target_folder}'):

        if frame_img not in img_to_coords:
            continue # skip no basket frames

        else:
            frame_img_path = os.path.join(frames_folder, frame_img)
            x1, y1, x2, y2 = img_to_coords[frame_img]

            if standardise:
                x_mid = (x1 + x2) // 2
                y_mid = (y1 + y2) // 2
                x1, x2 = x_mid - (crop_size // 2), x_mid + (crop_size // 2)
                y1, y2 = y_mid - (crop_size // 2), y_mid + (crop_size // 2)

            pil_img = Image.open(frame_img_path)
            cropped_pil_img = pil_img.crop((x1, y1, x2, y2))
            save_img_path = os.path.join(target_folder, frame_img)
            cropped_pil_img.save(save_img_path)
