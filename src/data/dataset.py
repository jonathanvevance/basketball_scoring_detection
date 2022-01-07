
import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset

from utils.file_utils import listdir
from utils.file_utils import get_empty_subdirectories
from utils.img_video_utils import filter_images_func


class binary_mil_folder(Dataset):
    """ Stack +ve and -ve bags here itself. Pad the two bags individually.
        In fwd function, we will apply forward() on the tensor consisting of the two bags.
        Output will be a tensor of scores. We will handle the bags in 'criterion'.
    """

    def __init__(self, dataset_path, transform = None, max_video_frames = 60):

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
        return len(self.scoring_videopaths)

    def __getitem__(self, idx):

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

    def __init__(self, dataset_path, transform, max_video_frames):

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

        num_scoring_undetected = len(self.scoring_basket_undetected_videopaths)
        num_nonscoring_undetected = len(self.nonscoring_basket_undetected_videopaths)
        augment_scores = [0 for __ in range(num_scoring_undetected + num_nonscoring_undetected)]
        augment_labels = [1 for __ in range(num_scoring_undetected)] + [0 for __ in range(num_nonscoring_undetected)]
        undetected_paths = self.scoring_basket_undetected_videopaths + self.nonscoring_basket_undetected_videopaths

        return augment_scores, augment_labels, undetected_paths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

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
        x = self.tensors[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.tensors)
