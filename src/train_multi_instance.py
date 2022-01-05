import os
import torch
from tqdm import tqdm
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader

import config as cfg
from models.conv_net import simpleConvNet
from data.dataset import binary_mil_folder
from utils.mil_utils import mil_loss
from utils.mil_utils import mil_model_wrapper
from utils.train_utils import save_model
from utils.train_utils import load_model
from utils.train_utils import EarlyStopping
from utils.eval_utils import evaluate_model
from utils.eval_utils import print_classification_metrics


def train():

    # load train/test transforms
    train_transform = transforms.Compose([
        transforms.Resize((cfg.RESIZE, cfg.RESIZE)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((cfg.RESIZE, cfg.RESIZE)),
        transforms.ToTensor(),
    ])

    # load dataset objects, data loaders
    train_dataset = binary_mil_folder(os.path.join(cfg.DATASET_FOLDER, "train"), transform = train_transform, max_video_frames = cfg.MAX_VIDEO_FRAMES)
    val_dataset = binary_mil_folder(os.path.join(cfg.DATASET_FOLDER, "val"), transform = test_transform, max_video_frames = cfg.MAX_VIDEO_FRAMES)

    train_loader = DataLoader(dataset = train_dataset, batch_size = cfg.BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = cfg.BATCH_SIZE, shuffle = True)

    # Strictly use GPU if available
    cfg.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = simpleConvNet(train_loader)
    model = model.to(cfg.DEVICE)
    model = mil_model_wrapper(model)

    # load saved weights if needed
    if cfg.LOAD_MODEL:
        load_model(model, cfg.LOAD_MODEL_PTH, cfg.DEVICE)

    # load criterion, optimizer
    criterion = mil_loss(cfg.MAX_VIDEO_FRAMES, cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = cfg.LEARNING_RATE, weight_decay = cfg.WEIGHT_DECAY)

    model = train_model(model, criterion, optimizer, train_loader, val_loader, cfg.EPOCHS, cfg.ES_PATIENCE, cfg.ES_DELTA, cfg.DEVICE)

    # saving the model
    save_model(model, cfg.SAVE_MODEL_PTH)

    # evaluate model
    print_classification_metrics(
        model,
        os.path.join(cfg.DATASET_FOLDER, 'test'),
        test_transform,
        cfg.MAX_VIDEO_FRAMES,
        cfg.BATCH_SIZE,
        cfg.DEVICE
    )


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, es_patience, es_delta, device):

    epoch = 0
    early_stop = False
    early_stopping = EarlyStopping(patience = es_patience , delta = es_delta)

    while (epoch < epochs) and not early_stop:

        model.train()
        train_losses_total = 0

        with tqdm(train_loader, unit = "batch", leave = True) as tqdm_progressbar:
            for idx, (inputs, labels) in enumerate(tqdm_progressbar):

                tqdm_progressbar.set_description(f"Epoch {epoch} (training)")

                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)

                optimizer.zero_grad()
                outputs = model(inputs).to(torch.float32)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_losses_total += loss.item()
                train_losses_avg = train_losses_total / (idx + 1)
                tqdm_progressbar.set_postfix(train_loss = train_losses_avg)

        if (epoch + 1) % 5 == 0:

            val_losses_avg = evaluate_model(model, criterion, val_loader, device, epoch)
            stop = early_stopping(val_losses_avg, model)

            if stop:
                print('INFO: Early stopped - val_loss_min: {}'.format(early_stopping.val_loss_min.item()))
                model.load_state_dict(early_stopping.saved_state_dict)
                early_stop = True

        epoch += 1

    return model

train()