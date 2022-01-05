
import os
import torch
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class binary_mil_folder(Dataset):
    """ Stack +ve and -ve bags here itself. Pad the two bags individually.
        In fwd function, we will apply forward() on the tensor consisting of the two bags.
        Output will be a tensor of scores. We will handle the bags in 'criterion'.
    """

    def __init__(self, dataset_path, transform = None, max_video_frames = 60):

        self.scoring_path = os.path.join(dataset_path, '1')
        self.nonscoring_path = os.path.join(dataset_path, '0')

        self.scoring_videos = os.listdir(self.scoring_path)
        self.nonscoring_videos = os.listdir(self.nonscoring_path)

        self.transform = transform
        self.max_video_frames = max_video_frames

    def __len__(self, ):
        return len(self.scoring_videos)

    def __getitem__(self, idx):

        scoring_video = self.scoring_videos[idx]
        scoring_video_path = os.path.join(self.scoring_path, scoring_video)

        # sample a random nonscoring video
        nonscoring_idx = random.randint(0, len(self.nonscoring_videos) - 1)
        nonscoring_video = self.nonscoring_videos[nonscoring_idx]
        nonscoring_video_path = os.path.join(self.nonscoring_path, nonscoring_video)

        bag_tensors = []
        scoring_frames = os.listdir(scoring_video_path)
        nonscoring_frames = os.listdir(nonscoring_video_path)

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


class video_folder(Dataset):

    def __init__(self, dataset_path, transform, max_video_frames):

        self.scoring_path = os.path.join(dataset_path, '1')
        self.nonscoring_path = os.path.join(dataset_path, '0')
        self.scoring_videopaths = [os.path.join(self.scoring_path, video) for video in os.listdir(self.scoring_path)]
        self.nonscoring_videopaths = [os.path.join(self.nonscoring_path, video) for video in os.listdir(self.nonscoring_path)]

        self.video_paths = self.scoring_videopaths + self.nonscoring_videopaths
        self.labels = [1 for __ in range(len(self.scoring_videopaths))] + [0 for __ in range(len(self.nonscoring_videopaths))]

        self.transform = transform
        self.max_video_frames = max_video_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        bag_tensors = []
        video_path = self.video_paths[idx]
        frame_paths = [os.path.join(video_path, frame) for frame in os.listdir(video_path)]

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

        return torch.stack(bag_tensors), torch.Tensor([self.labels[idx], self.max_video_frames - pad_length])


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
        return self.tensors[0].size(0)
