"""Util functions for evaluating models."""

import os
import csv
import json
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

from data.dataset import eval_folder
from models.conv_net import simpleConvNet
from utils.mil_utils import mil_model_wrapper


def load_model_clf(MODEL_WEIGHTS_PATH, device):
    """
    Saved weights are for utils.mil_utils.mil_model_wrapper
    class objects. Here we load them into a suitable class object
    and then recover the model.model classification model inside.
    """
    model = simpleConvNet()
    model = mil_model_wrapper(model.to(device))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location = device))
    return model.model


class get_batch_video_probabs(nn.Module):
    """Get output scores and labels of a batch of videos."""
    def __init__(self, device, max_video_frames):
        super().__init__()
        self.device = device
        self.max_video_frames = max_video_frames

    def forward(self, outputs, labels, len_frames):
        frames_mask = torch.arange(self.max_video_frames, device = self.device)[None, :] < len_frames
        output_masked = outputs * frames_mask.int().float()
        video_probab_scores = torch.amax(output_masked, dim = 1)

        return video_probab_scores.tolist(), labels[:, 0].tolist()


def get_all_video_probabs(model, dataloader, device, max_video_frames):
    """Get output scores and labels of all videos in the dataloader using batch evaluation."""

    model.eval()
    all_paths = []
    all_labels = []
    all_scores = []
    criterion = get_batch_video_probabs(device, max_video_frames)

    with tqdm(dataloader, unit = "batch", leave = True) as tqdm_progressbar:
        for batch in tqdm_progressbar:
            inputs, labels, vidlens, vidpaths = batch['X'], batch['Y'], batch['len'], batch['path']

            tqdm_progressbar.set_description(f"Getting video predictions")

            inputs, labels, vidlens = inputs.to(device), labels.to(device), vidlens.to(device)
            inputs, labels, vidlens = inputs.to(torch.float32), labels.to(torch.float32), vidlens.to(torch.float32)

            outputs = model(inputs).to(torch.float32)
            batch_preds, batch_labels = criterion(outputs, labels, vidlens)

            all_paths.extend(vidpaths)
            all_scores.extend(batch_preds)
            all_labels.extend(batch_labels) # ADDED

    return all_scores, all_labels, all_paths


def get_optimal_threshold(model, val_path, transform, max_video_frames, batch_size, device):
    """Returns optimal threshold based on (TPR - FPR) score on validation set."""

    # get dataset, dataloader
    dataset = eval_folder(val_path, transform, max_video_frames)
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    # criteria
    scores, labels, paths = get_all_video_probabs(model, dataloader, device, max_video_frames)

    # account for (basket) undetected videos (tjese wo;; )
    augment_scores, augment_labels, augment_paths = dataset.get_basket_undetected_videos()
    scores, labels, paths = scores + augment_scores, labels  + augment_labels, paths + augment_paths

    # decide optimal threshold (TPR - FPR)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold


def print_classification_metrics(
    model, test_path, transform, max_video_frames, batch_size, device, save_results_dir = None, threshold = None, val_path = None
):
    """Prints and saves classification metrics - ROC curve, AUROC score, predictions as csv file etc."""

    # get dataset, dataloader
    dataset = eval_folder(test_path, transform, max_video_frames)
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    # criteria
    scores, labels, paths = get_all_video_probabs(model, dataloader, device, max_video_frames)

    # account for (basket) undetected videos (tjese wo;; )
    augment_scores, augment_labels, augment_paths = dataset.get_basket_undetected_videos()
    scores, labels, paths = scores + augment_scores, labels  + augment_labels, paths + augment_paths

    # show ROC curve
    auroc_score = roc_auc_score(labels, scores)
    print("AUROC score is", auroc_score)
    RocCurveDisplay.from_predictions(labels, scores)
    plt.show()

    if save_results_dir:
        plt.savefig(os.path.join(save_results_dir, 'roc_curve.jpg'))

    if threshold is None: # when calling from src/train.py
        threshold = get_optimal_threshold(model, val_path, transform, max_video_frames, batch_size, device)

    # get classification report
    print("Threshold is", threshold)
    y_pred_class = list(map(int, np.array(scores) > threshold))
    report = classification_report(labels, y_pred_class)
    print(report)

    # save classification report
    if save_results_dir:

        results_dict = {
            'dataset': test_path,
            'auroc_score': auroc_score,
            'threshold': threshold,
            'classification_report': report
        }
        with open(os.path.join(save_results_dir, 'results.json'), 'w') as f:
            json.dump(results_dict, f)

        # save prediction csv
        predictions_rows = list(zip(paths, scores, y_pred_class, labels))
        with open(os.path.join(save_results_dir, "predictions.csv"), 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'score', 'prediction', 'target'])
            writer.writerows(predictions_rows)

        print(f"\nResults saved in {save_results_dir}")
