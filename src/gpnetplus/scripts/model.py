from builtins import super

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from mmseg.apis import init_model


class FcnResnet50(nn.Module):
    """
    GP-net+ model with a Fully Convolutional ResNet50 architecture.
    """
    def __init__(self):
        super(FcnResnet50, self).__init__()

        num_classes = 6
        self.model = models.segmentation.fcn_resnet50(num_classes=num_classes, weights=None)

    def forward(self, x):
        y = self.model(x)['out']

        qual_out = torch.sigmoid(y[:, 0:1, :, :])
        rot_out = F.normalize(y[:, 1:5, :, :], dim=1)
        width_out = torch.sigmoid(y[:, 5:, :, :])
        return qual_out, rot_out, width_out

def load_network(path, device, type=None):
    """Construct the neural network and load parameters from the specified file.

    Args:
        path: Path to the model parameters. The name must conform to `vgn_name_[_...]`.
    """
    if type == 'resnet':
        net = FcnResnet50().to(device)
    else:
        raise AttributeError("No architecture type given in model.py")
    try:
        net.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError:
        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(path, map_location=device))
        net = net.module
    net.eval()
    return net


