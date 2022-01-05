from utils.img_video_utils import get_cropped_pil_images_inference
from utils.img_video_utils import save_frames_from_video_inference
import os

def run_predictions(pil_images, frame_level_required):
    #! TODO: to be coded
    return False, [0.5 for __ in range(len(pil_images))]


def predict(frame_level_required = True):
    '''predict Yes or No per frame per video.'''

    # get frames from video
    save_frames_from_video_inference()

    # run basket detection
    os.system('''
        cd ./src/yolov3_helper
        sudo bash setup.sh
    ''') #! to be changed to predict.sh (see make_dataset_*)
    # bounding boxes saved to src/yolov3_helper/yolov3/output/bounding_boxes.json

    # crop basket images
    pil_images = get_cropped_pil_images_inference()

    # run predictions
    is_scoring, pred_probabs = run_predictions(pil_images, frame_level_required)

    return is_scoring, pred_probabs


predict()