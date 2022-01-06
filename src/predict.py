import os
import csv
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.img_video_utils import get_cropped_pil_images_inference
from utils.img_video_utils import save_frames_from_video_inference
from utils.eval_utils import load_model_clf
from data.dataset import CustomTensorDataset

RESIZE = 128
THRESHOLD = 0.93
BATCH_SIZE = 128 #! TODO: REQUEST BATCH SIZE AS ENV VAR
MODEL_WEIGHTS_PATH = 'models/best.pt'
FRAMES_UPLOAD_DIRECTORY = 'data/inference/frames_upload'
FRAMES_PROBAB_CSV_PATH = 'reports/probability_values.csv'


def run_predictions_batch(pil_images):

    is_scoring = False
    frame_probabs = [['time', 'values']]

    is_valid_mask = [True if img else False for img in pil_images]
    pil_images = [img for img in pil_images if img is not None]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = load_model_clf(MODEL_WEIGHTS_PATH, device)
    model = model.eval()

    transform = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor()
    ])

    dataset = CustomTensorDataset(pil_images, transform)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)

    all_valid_probabs = []
    for img_batch in tqdm(dataloader):
        img_batch = img_batch.to(device)
        probabs_batch = model(img_batch)
        all_valid_probabs.extend(torch.squeeze(probabs_batch).tolist())

    print(all_valid_probabs)

    valid_idx = 0
    for frame_num, is_valid in enumerate(is_valid_mask):
        if is_valid:
            if all_valid_probabs[valid_idx] > THRESHOLD:
                is_scoring = True
            frame_probabs.append([frame_num, all_valid_probabs[valid_idx]])
            valid_idx += 1
        else:
            frame_probabs.append([frame_num, 0])

    return is_scoring, frame_probabs


def run_predictions(pil_images):

    is_scoring = False
    frame_probabs = [['time', 'values']]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = load_model_clf(MODEL_WEIGHTS_PATH, device)
    model = model.eval()

    # run predictions, return predictions
    for frame_num, pil_image in enumerate(tqdm(pil_images)):

        if pil_image is None: # no basket detected
            frame_probabs.append([frame_num, 0])
            continue

        pil_image = transforms.Resize((RESIZE, RESIZE))(pil_image)
        image_tensor = transforms.ToTensor()(pil_image)
        image_tensor = image_tensor.to(device)
        output = model(torch.unsqueeze(image_tensor, 0))
        frame_probabs.append([frame_num, output[0].item()])

        if output > THRESHOLD:
            is_scoring = True

    return is_scoring, frame_probabs


def predict():
    """predict Yes or No per frame per video."""

    # get frames from video
    save_frames_from_video_inference()

    # run basket detection
    commands = [
        "cd ./src/yolov3_helper",
        "sudo bash predict.sh" + " ../../../" + FRAMES_UPLOAD_DIRECTORY + " ../../../" + FRAMES_UPLOAD_DIRECTORY,
    ]
    os.system(";".join(commands))  # bounding boxes saved to src/yolov3_helper/yolov3/output/bounding_boxes.json

    # crop basket images
    pil_images = get_cropped_pil_images_inference()

    # run predictions
    is_scoring, pred_probabs = run_predictions(pil_images)
    # is_scoring, pred_probabs = run_predictions_batch(pil_images)

    # save as csv file
    with open(FRAMES_PROBAB_CSV_PATH, 'w', newline = "") as f:
        writer = csv.writer(f)
        writer.writerows(pred_probabs)

    return is_scoring

    #! TODO: SEE PROBABS FROM SAVED_MODEL_6.PT - are they good?


# predict()
