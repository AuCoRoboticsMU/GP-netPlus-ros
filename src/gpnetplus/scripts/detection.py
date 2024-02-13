import time

import numpy as np
import torch
from transform import quaternion_rotation_matrix

from utils import Grasp, depth_encoding
from model import load_network
from skimage.feature import peak_local_max

class GPnetPlus(object):
    def __init__(self, model_path, rviz=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device, type='resnet')
        self.rviz = rviz
        self.depth_min = 0.2
        self.depth_max = 1.5
        self.width_max = 0.08

    def __call__(self, depth_im, K):
        """
        Run a single forward pass with GP-net+, select grasp proposals using Non-Maximum-Suppression, map them to 3D
        coordinates and sort the resulting grasps based on their predicted grasp confidence.

        :param depth_im: Depth image that is being used as input to GP-net+
        :param K: Intrinsic parameters for the camera, used to deproject the selected grasps to 3D
        """
        tic = time.time()
        qual_pred, rot_pred, width_pred, jet_im = self.predict(depth_im)

        # Set qualities for grasps that are not visible to 0.
        qual_pred = np.array(qual_pred).squeeze()
        qual_pred[np.where(depth_im == 0.0)] = 0.0
        # Select grasps using NMS
        grasps, scores, z_dir, height, coordinates = self.select_grasps(qual_pred.copy(), rot_pred, width_pred, depth_im, K)
        toc = time.time() - tic

        # Sort grasps based on their predicted grasp confidence
        grasps, scores, z_dir, height = np.asarray(grasps), np.asarray(scores), np.asarray(z_dir), np.asarray(height)
        coordinates = np.asarray(coordinates)
        indices = scores.argsort()[::-1]

        return grasps[indices], scores[indices], coordinates[indices], toc

    def predict(self, depth_image):
        """
        Convert a depth image to a jet-colourscale and then use it to run a single forward pass with GP-net+.
        """
        jet_im = depth_encoding(depth_image, vmin=self.depth_min, vmax=self.depth_max)

        x = torch.from_numpy(jet_im).unsqueeze(0).to(self.device)

        # forward pass
        with torch.no_grad():
            qual_vol, rot_vol, width_vol = self.net(x)

        # move output back to the CPU
        qual_vol = qual_vol.cpu().detach().numpy()
        rot_vol = rot_vol.cpu().detach().numpy()
        width_vol = width_vol.cpu().detach().numpy()
        return qual_vol, rot_vol, width_vol, jet_im

    def select_grasps(self, pred_qual, pred_quat, pred_width, depth_im, K, threshold=0.29):
        """
        Select grasp proposals using NMS and project them into 3D coordinates
        """
        indices = peak_local_max(pred_qual.squeeze(), min_distance=4, threshold_abs=threshold)
        grasps = []
        qualities = []
        poses = []
        heights = []

        for index in indices:
            quaternion = pred_quat.squeeze()[:, index[0], index[1]]
            quality = pred_qual.squeeze()[index[0], index[1]]

            # Relative width, so multiply with maximum width of gripper
            width = pred_width.squeeze()[index[0], index[1]] * self.width_max

            contact = (index[1], index[0])
            grasp, T_camera_tcp = self.reconstruct_grasp_from_variables(depth_im, contact, quaternion, width, K)

            if grasp is None:
                continue

            grasps.append(grasp)
            qualities.append(quality)
            poses.append(T_camera_tcp[2, 0])
            heights.append(T_camera_tcp[2, 3])
        return grasps, qualities, poses, heights, indices

    def reconstruct_grasp_from_variables(self, depth_im, contact, quaternion, width, K):
        # Deproject from depth image into image coordinates
        # Note that homogeneous coordinates have the image coordinate order (x, y), while accessing the depth image
        # works with numpy coordinates (row, column)
        homog = np.array((contact[0], contact[1], 1)).reshape((3, 1))
        if depth_im[contact[1], contact[0]] == 0.0:
            return None, None
        point = depth_im[contact[1], contact[0]] * np.linalg.inv(np.array(K).reshape(3, 3)).dot(homog)
        point = point.squeeze()

        # Transform the quaternion into a rotation matrix
        rot = quaternion_rotation_matrix(quaternion)

        # Move from contact to grasp centre by traversing 0.5*grasp width in grasp axis direction
        centre_point = point + width / 2 * rot.T[0, :]

        # Construct transform Camera --> gripper
        T_camera_tcp = np.r_[np.c_[rot, centre_point], [[0, 0, 0, 1]]]

        return Grasp(T_camera_tcp, width), T_camera_tcp


