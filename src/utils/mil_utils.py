"""MIL functions."""

import torch
import torch.nn as nn

class mil_loss(nn.Module):
    """
    The loss function = -[max(scores of positive bag)) - max(scores of negative bag)].
    """
    def __init__(self, max_video_frames, device):
        """Init function."""
        super().__init__()
        self.max_video_frames = max_video_frames
        self.device = device

    def forward(self, outputs, labels):
        """
        Outputs: Batches of paired scoring - non scoring frames (B, NF, C, H, W)
        Inputs : Batches of num. frames of each pair of scoring - non scoring video (before zero padding).
        -> For more information, see src/data/dataset.py for documentation of binary_mil_folder

        Steps:
        - Split output into scoring and non scoring, labels into lengths of scoring and non scoring videos.
        - Mask out the outputs from the zero padded parts of the video batches (ideas from masking in RNNs)
        - Take the max of outputs (probabilities) across the frames dimension (dim = 1) apply loss function.

        Note: an additional 1.00001 is added so that the loss is always > 0.
        """

        len_scoring_batch, len_nonscoring_batch = torch.split(labels, 1, dim = 1)
        scoring_output_batch, nonscoring_output_batch = torch.split(outputs, self.max_video_frames, dim = 1)

        # https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
        scoring_mask = torch.arange(self.max_video_frames, device = self.device)[None, :] < len_scoring_batch
        scoring_output_masked = scoring_output_batch * scoring_mask.int().float()

        nonscoring_mask = torch.arange(self.max_video_frames, device = self.device)[None, :] < len_nonscoring_batch
        nonscoring_output_masked = nonscoring_output_batch * nonscoring_mask.int().float()

        scoring_max_probab = torch.amax(scoring_output_masked, dim = 1)
        nonscoring_max_probab = torch.amax(nonscoring_output_masked, dim = 1)

        return - (torch.mean(scoring_max_probab) - torch.mean(nonscoring_max_probab)) + 1.00001 # +1 so that loss >= 0


class mil_model_wrapper(nn.Module):
    """
    MIL model wrapper for the classification model being trained.

    Why is it required?
    - The inputs come in 5D tensors (see src/data/dataset.py for details).
    - Covolution operation requires inputs to come in 4D input tensors.

    What is being done?
    - Thus we reshape the 5D tensor into a 4D tensor, pass it through the model in the
    forward() and reshape the outputs (one output unit only) to a 2D output (B, NF).
    - We are basically tricking the model to thinking that the data is a 4D tensor.
    """
    def __init__(self, model):
        """Init function."""
        super().__init__()
        self.model = model

    def forward(self, x):
        """Run forward propagation after reshaping operation."""
        batch_size, total_frames, *last_dims = list(x.shape)
        x = torch.reshape(x, [batch_size * total_frames] + last_dims)
        output = torch.reshape(self.model(x), [batch_size, total_frames])
        return output
