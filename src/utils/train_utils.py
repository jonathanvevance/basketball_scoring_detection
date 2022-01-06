# Inspired by: https://github.com/jakubmonhart/mil_pytorch/blob/71c7f7183025dcb79da25fef67289d71c6209ce8/mil_pytorch/utils/train_utils.py

import torch
from tqdm import tqdm

class EarlyStopping():
    def __init__(self, patience = 10, delta = 0):
        self.patience = patience
        self.delta = delta
        self.val_loss_min = None
        self.saved_state_dict = None
        self.counter = 0

    def __call__(self, val_loss, model):
        if self.val_loss_min is None:
            self.val_loss_min = val_loss
            self.saved_state_dict = model.state_dict()
            return False

        change = (self.val_loss_min - val_loss) / self.val_loss_min

        if change >= self.delta:
            self.counter = 0
            self.val_loss_min = val_loss
            self.saved_state_dict = model.state_dict()
            return False
        else:
            self.counter += 1

            if self.counter > self.patience:
                return True
            else:
                return False


def evaluate_model(model, criterion, val_loader, device, epoch):
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


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, load_path, device):
    model.load_state_dict(torch.load(load_path, map_location = device))
    return model