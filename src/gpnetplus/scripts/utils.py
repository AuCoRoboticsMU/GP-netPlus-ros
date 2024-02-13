import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def depth_encoding(image, vmin=0.2, vmax=2.5):
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colored_image = plt.cm.jet(norm(image))[:, :, :-1]

    return np.array((colored_image[:, :, 0], colored_image[:, :, 1], colored_image[:, :, 2]), dtype=np.float32)

class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    """

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width
