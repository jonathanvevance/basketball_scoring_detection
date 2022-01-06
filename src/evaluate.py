

from configs import eval_config as cfg
from models.conv_net import simpleConvNet
from utils.mil_utils import mil_model_wrapper
from utils.train_utils import load_model
from utils.eval_utils import print_classification_matrics

#!!! HAVE TO CONVERT VIDEOS TO CROPPED FRAMES (use make dataset WITH OPTIONS)

def evaluate():

    # load model
    model = simpleConvNet(train_loader)
    model = model.to(cfg.DEVICE)
    model = mil_model_wrapper(model)

    # load saved weights if needed
    if cfg.LOAD_MODEL:
        load_model(model, cfg.LOAD_MODEL_PTH, cfg.DEVICE)

    print_classification_metrics(model, dataset_path, transform, max_video_frames, batch_size, device, threshold)