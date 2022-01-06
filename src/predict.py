
import os
import torch
from torchvision import transforms

from utils.img_video_utils import get_cropped_pil_images_inference
from utils.img_video_utils import save_frames_from_video_inference
from utils.eval_utils import load_model_clf

RESIZE = 128
THRESHOLD = 0.8
MODEL_WEIGHTS_PATH = 'models/best.pt'
FRAMES_UPLOAD_DIRECTORY = 'data/inference/frames_upload'

def run_predictions(pil_images):

    is_scoring = False
    frame_probabs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = load_model_clf(MODEL_WEIGHTS_PATH, device)
    model = model.eval()

    # run predictions, return predictions
    for i, pil_image in enumerate(pil_images):

        if pil_image is None: # no basket detected
            frame_probabs.append(0)
            continue

        pil_image = transforms.Resize((RESIZE, RESIZE))(pil_image)
        image_tensor = transforms.ToTensor()(pil_image)
        output = model(torch.unsqueeze(image_tensor, 0))
        print(i, output)
        frame_probabs.append(output)

        if output > THRESHOLD:
            is_scoring = True

    return is_scoring, frame_probabs


def predict():
    '''predict Yes or No per frame per video.'''

    # get frames from video
    save_frames_from_video_inference()

    # run basket detection
    commands = [
        'cd ./src/yolov3_helper',
        'sudo bash predict.sh' + ' ../../../' + FRAMES_UPLOAD_DIRECTORY + ' ../../../' + FRAMES_UPLOAD_DIRECTORY
    ]
    os.system(';'.join(commands)) # bounding boxes saved to src/yolov3_helper/yolov3/output/bounding_boxes.json

    # crop basket images
    pil_images = get_cropped_pil_images_inference()

    # run predictions
    is_scoring, pred_probabs = run_predictions(pil_images)

    return is_scoring, pred_probabs
    #! CONFIRM FOMMAT. MAKE INTO CSV FILE (RETURN OR SAVE - ASK ADHIL)

    #! SEE PROBABS FROM SAVED_MODEL_6.PT - are they good enough?

    #! 

predict()