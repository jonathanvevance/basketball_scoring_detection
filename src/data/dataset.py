"""Classes to load datasets."""

import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset

from utils.file_utils import listdir
from utils.file_utils import get_empty_subdirectories
from utils.img_video_utils import filter_images_func


class binary_mil_folder(Dataset):
    """
    Two-class MIL folder.
    - Gives 5D tensors of the shape (B, NF, C, H, W) where NF = 2x the max num of frames
    of each video (in the given dataset, 2 x 60 = 120).
    - Each tensor contains frames of a pair of scoring and non scoring video (hence 2x).
    - Zero padding is done on both videos (individually) so that the NF=120 is maintained,
    in case there are less than 60 frames in any video.
    """
    def __init__(self, dataset_path, transform = None, max_video_frames = 60):
        """Init function."""
        self.scoring_path = os.path.join(dataset_path, '1')
        self.nonscoring_path = os.path.join(dataset_path, '0')

        # filter out empty video folders
        __, valid_scoring_folders = get_empty_subdirectories(self.scoring_path)
        __, valid_nonscoring_folders = get_empty_subdirectories(self.nonscoring_path)

        self.scoring_videopaths = valid_scoring_folders
        self.nonscoring_videopaths = valid_nonscoring_folders

        self.transform = transform
        self.max_video_frames = max_video_frames

    def __len__(self, ):
        """Returns number of scoring videos."""
        return len(self.scoring_videopaths)

    def __getitem__(self, idx):
        """
        Implementation detail:
            The dataloader is built on list of scoring videos. When the scoring video
        specified by 'idx' is requested in __getitem__, a random index for non scoring
        video is sampled and the sampled non scoring video is paired with the requested
        non scoring video. The frames from this pair of videos are (torch) stacked.

            It is known that the first 60 frames (with appropriate padding) will be from
        a scoring video and the last 60 will be from the non scoring video. But different
        videos may have different number of frames available - this is because, only those
        frames from which the basketball hoop has been located, ends up in the final MIL
        training dataset. So instead of any labels, we send the information of the number
        of frames from the scoring and the non scoring video, before zero padding.
        """
        scoring_video_path = self.scoring_videopaths[idx]

        # sample a random nonscoring video
        nonscoring_idx = random.randint(0, len(self.nonscoring_videopaths) - 1)
        nonscoring_video_path = self.nonscoring_videopaths[nonscoring_idx]

        bag_tensors = []
        scoring_frames = list(filter(filter_images_func, listdir(scoring_video_path)))
        nonscoring_frames = list(filter(filter_images_func, listdir(nonscoring_video_path)))

        for frame in scoring_frames:
            pil_img = Image.open(os.path.join(scoring_video_path, frame))

            if self.transform is not None:
                img_tensor = self.transform(pil_img)
            else:
                img_tensor = pil_img

            bag_tensors.append(img_tensor)

        # padding the scoring part
        scoring_pad_length = self.max_video_frames - len(bag_tensors)
        for __ in range(scoring_pad_length):
            bag_tensors.append(torch.zeros_like(bag_tensors[-1]))

        for frame in nonscoring_frames:
            pil_img = Image.open(os.path.join(nonscoring_video_path, frame))

            if self.transform is not None:
                img_tensor = self.transform(pil_img)
            else:
                img_tensor = pil_img

            bag_tensors.append(img_tensor)

        # padding the nonscoring part
        nonscoring_pad_length = ( 2 * self.max_video_frames) - len(bag_tensors)
        for __ in range(nonscoring_pad_length):
            bag_tensors.append(torch.zeros_like(bag_tensors[-1]))

        return torch.stack(bag_tensors), torch.tensor([len(scoring_frames), len(nonscoring_frames)])


class eval_folder(Dataset):
    """
    Evaluation (video) folder.
    - Gives 5D tensors of the shape (B, NF, C, H, W) where NF = max num of frames of each
    video (in the given dataset = 60).
    - Each tensor contains frames of a scoring or a non scoring video.
    - Zero padding is done so that the NF=60 is maintained, in case there are less than 60
    frames in the requested video video.
    """
    def __init__(self, dataset_path, transform, max_video_frames):
        """Init function."""
        self.scoring_path = os.path.join(dataset_path, '1')
        self.nonscoring_path = os.path.join(dataset_path, '0')

        # filter out empty video folders
        empty_scoring_folders, valid_scoring_folders = get_empty_subdirectories(self.scoring_path)
        empty_nonscoring_folders, valid_nonscoring_folders = get_empty_subdirectories(self.nonscoring_path)

        self.scoring_videopaths = valid_scoring_folders
        self.nonscoring_videopaths = valid_nonscoring_folders
        self.scoring_basket_undetected_videopaths = empty_scoring_folders
        self.nonscoring_basket_undetected_videopaths = empty_nonscoring_folders

        self.video_paths = self.scoring_videopaths + self.nonscoring_videopaths
        self.labels = [1 for __ in range(len(self.scoring_videopaths))] + [0 for __ in range(len(self.nonscoring_videopaths))]

        self.transform = transform
        self.max_video_frames = max_video_frames

    def get_basket_undetected_videos(self):
        """
            In the evaluation folder, there can be subfolders for videos where the yolov3 basket
        detector was not able to find any basket frames. This will result in empty folders for those
        videos. We filter out these rows in advance because the dataloader may be requested to return
        fully zero padded tensors otherwise.

        At inference, these videos will be predicted as 0. So this function returns scores (all 0s), labels
        (and paths) of these videos which will be used to augment the predictions and labels from non empty
        folders.
        """
        num_scoring_undetected = len(self.scoring_basket_undetected_videopaths)
        num_nonscoring_undetected = len(self.nonscoring_basket_undetected_videopaths)
        augment_scores = [0 for __ in range(num_scoring_undetected + num_nonscoring_undetected)]
        augment_labels = [1 for __ in range(num_scoring_undetected)] + [0 for __ in range(num_nonscoring_undetected)]
        undetected_paths = self.scoring_basket_undetected_videopaths + self.nonscoring_basket_undetected_videopaths

        return augment_scores, augment_labels, undetected_paths

    def __len__(self):
        """Returns number of videos."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns dictionary with key-values
            X: data tensor
            Y: lebels tensor
            len: number of valid frames (without zero padding)
            path: path of the video (for debugging, saving results)
        """

        bag_tensors = []
        video_path = self.video_paths[idx]
        frames = list(filter(filter_images_func, listdir(video_path)))
        frame_paths = [os.path.join(video_path, frame) for frame in frames]

        for frame_path in frame_paths:
            pil_img = Image.open(frame_path)

            if self.transform is not None:
                img_tensor = self.transform(pil_img)
            else:
                img_tensor = pil_img

            bag_tensors.append(img_tensor)

        pad_length = self.max_video_frames - len(bag_tensors)
        for __ in range(pad_length):
            bag_tensors.append(torch.zeros_like(bag_tensors[-1]))

        return {
            'X': torch.stack(bag_tensors),
            'Y': torch.Tensor([self.labels[idx]]),
            'len': torch.Tensor([self.max_video_frames - pad_length]),
            'path': video_path
        }


# https://stackoverflow.com/a/55593757
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        """Returns tensor at index."""
        x = self.tensors[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        """Returns number of tensors."""
        return len(self.tensors)
