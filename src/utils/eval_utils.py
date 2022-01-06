
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from data.dataset import video_folder
from models.conv_net import simpleConvNet
from utils.mil_utils import mil_model_wrapper


def load_model_clf(MODEL_WEIGHTS_PATH, device):
    model = simpleConvNet()
    model = mil_model_wrapper(model.to(device))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location = device))
    return model.model


class get_batch_video_probabs(nn.Module):
    def __init__(self, device, max_video_frames):
        super().__init__()
        self.device = device
        self.max_video_frames = max_video_frames

    def forward(self, outputs, labels):
        labels_batch, len_frames_batch = torch.split(labels, 1, dim = 1)
        frames_mask = torch.arange(self.max_video_frames, device = self.device)[None, :] < len_frames_batch
        output_masked = outputs * frames_mask.int().float()
        video_probab_scores = torch.amax(output_masked, dim = 1)

        return video_probab_scores.tolist(), labels_batch[:, 0].tolist()


def get_all_video_probabs(model, dataloader, device, max_video_frames):

    model.eval()
    all_labels = []
    all_scores = []
    criterion = get_batch_video_probabs(device, max_video_frames)

    with tqdm(dataloader, unit = "batch", leave = True) as tqdm_progressbar:
        for (inputs, labels) in tqdm_progressbar:

            tqdm_progressbar.set_description(f"Getting video predictions")

            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)

            outputs = model(inputs).to(torch.float32)
            batch_preds, batch_labels = criterion(outputs, labels)

            all_scores.extend(batch_preds)
            all_labels.extend(batch_labels)

    return all_scores, all_labels


def print_classification_metrics(model, dataset_path, transform, max_video_frames, batch_size, device, threshold = None):

    # get dataset, dataloader
    dataset = video_folder(dataset_path, transform, max_video_frames)
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    # criteria
    scores, labels = get_all_video_probabs(model, dataloader, device, max_video_frames)

    # show ROC curve
    auroc_score = roc_auc_score(labels, scores)
    print("\n---------------------------\n")
    print("AUROC SCORE ON THE DATASET IS :", auroc_score)
    RocCurveDisplay.from_predictions(labels, scores)
    plt.show()

    # decide optimal threshold (TPR - FPR)
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]

    # get classification report
    y_pred_class = scores > threshold
    print("\n---------------------------\n")
    print("Threshold value is:", threshold)
    print(classification_report(labels, y_pred_class))
