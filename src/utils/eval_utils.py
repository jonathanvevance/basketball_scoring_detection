
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from data.dataset import video_folder

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

    i = 0 #!

    with tqdm(dataloader, unit = "batch", leave = True) as tqdm_progressbar:
        for (inputs, labels) in tqdm_progressbar:

            i += 1
            if i == 3:
                break

            tqdm_progressbar.set_description(f"Getting video predictions")

            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)

            outputs = model(inputs).to(torch.float32)
            batch_preds, batch_labels = criterion(outputs, labels)

            all_scores.extend(batch_preds)
            all_labels.extend(batch_labels)

    return all_scores, all_labels


def evaluate_model(model, criterion, val_loader, device, epoch = -1):
    """
    model :: mil_model_wrapper object
    val_loader :: binary_mil_folder object
    criterion :: mil_loss object
    """

    model.eval()
    val_losses_total = 0

    with tqdm(val_loader, unit = "batch", leave = True) as tqdm_progressbar:
        for idx, (inputs, labels) in enumerate(tqdm_progressbar):

            tqdm_progressbar.set_description(f"Epoch {epoch} (validating)")

            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)

            outputs = model(inputs).to(torch.float32)
            loss = criterion(outputs, labels)

            val_losses_total += loss.item()
            val_losses_avg = val_losses_total / (idx + 1)
            tqdm_progressbar.set_postfix(val_loss = val_losses_avg)

    return val_losses_avg


def print_classification_metrics(model, dataset_path, transform, max_video_frames, batch_size, device):

    # get dataset, dataloader
    dataset = video_folder(dataset_path, transform, max_video_frames, )
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    # criteria
    scores, labels = get_all_video_probabs(model, dataloader, device, max_video_frames)

    print(scores)
    print(type(scores))
    print(len(scores))

    # print classification report
    auroc_score = roc_auc_score(labels, scores)

    print("\n---------------------------\n")
    print("AUROC SCORE ON THE DATASET IS :", auroc_score)
