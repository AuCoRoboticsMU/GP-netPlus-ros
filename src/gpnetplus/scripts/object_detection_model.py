from builtins import super
import numpy as np
from PIL import Image, ImageFont, ImageColor, ImageDraw

import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

plt.rcParams["savefig.bbox"] = 'tight'


def show(img):
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.imshow(img, aspect='auto')
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def _generate_color_palette(num_objects: int):
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    return [tuple((i * palette) % 255) for i in range(num_objects)]

def draw_bounding_boxes(img, labels, boxes, width=3, colors=None):
    ndarr = img.permute(1, 2, 0).cpu().numpy()
    pil_im = Image.fromarray(ndarr)

    img_boxes = boxes.to(torch.int64).tolist()
    num_boxes = len(img_boxes)

    if colors is None:
        colors = _generate_color_palette(num_boxes)
    elif isinstance(colors, list):
        if len(colors) < num_boxes:
            raise ValueError(f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). ")
    else:  # colors specifies a single color for all boxes
        colors = [colors] * num_boxes

    font = ImageFont.truetype("Ubuntu-Regular.ttf", 11)
    colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]
    if labels is not None and img_boxes is not None:
        draw = ImageDraw.Draw(pil_im)
        for bbox, color, label in zip(img_boxes, colors, labels):
            draw.rectangle(bbox, width=width, outline=color)
            margin = 5
            draw.text((bbox[0] + margin, bbox[1] + margin), label, font=font, color=(255, 255, 255))
    return pil_im

class FasterRCNN(torch.nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()

        self.model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.transforms = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
        self.model.eval()

    def forward(self, rgb_im, timestamp):
        # im = decode_image(torch.from_numpy(rgb_im))
        im = torch.from_numpy(rgb_im).permute(2, 0, 1)

        images = self.transforms(im)
        predictions = self.model([images])[0]

        labels = [FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"][label] for label in predictions['labels']]

        indices = torch.gt(predictions['scores'], 0.7)
        show(draw_bounding_boxes(im, np.array(labels)[indices.cpu().detach().numpy()],
                                 predictions['boxes'][indices, :], width=2))

        return predictions['boxes'].cpu().detach().numpy(), predictions['scores'].cpu().detach().numpy(), np.array(labels)



