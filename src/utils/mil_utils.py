
import torch
import torch.nn as nn

class mil_loss(nn.Module):
    def __init__(self, max_video_frames, device):
        super().__init__()
        self.max_video_frames = max_video_frames
        self.device = device

    def forward(self, outputs, labels):

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
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        batch_size, total_frames, *last_dims = list(x.shape)
        x = torch.reshape(x, [batch_size * total_frames] + last_dims)
        output = torch.reshape(self.model(x), [batch_size, total_frames])
        return output
