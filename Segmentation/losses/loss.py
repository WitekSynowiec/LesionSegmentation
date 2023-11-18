import torch
from torch import nn
from Segmentation.losses import functional as f


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8, alpha=0.5, beta=0.5):

        return f.tversky_loss(inputs, targets, smooth, alpha, beta)



if __name__ == "__main__":
    import numpy as np

    a = np.array([[1,0,1], [0, 1, 1]])
    b = np.array([[1,1,0], [0, 0, 0]])

    aa = torch.from_numpy(a)
    bb = torch.from_numpy(b)

