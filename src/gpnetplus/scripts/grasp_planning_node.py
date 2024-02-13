#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np

from detection import GPnetPlus
from transform import matrix_to_quaternion

from gpnetplus.srv import GPnetPlusGraspPlanner
from gpnetplus.msg import GPnetPlusGrasp, GraspProposals

class GraspPlanner(object):
    def __init__(self, cv_bridge, grasp_proposal_network):
        """
        Parameters
        cv_bridge: :obj:`CvBridge`
            ROS `CvBridge`.
        grasp_proposal_network: :obj:`GraspingPolicy`
            Grasping policy to use.
        """
        self.cv_bridge = cv_bridge
        self.gpnetplus = grasp_proposal_network

    def read_images(self, req):
        """Reads images from a ROS service request.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS ServiceRequest for grasp planner service.
        """
        # Get the raw depth image as ROS `Image` objects.
        raw_depths = req.depth_images

        # Get the raw camera info as ROS `CameraInfo`.
        raw_camera_info = req.camera_info

        # Unpacking the ROS depth image using ROS `CvBridge`
        depth_ims = []
        for raw_depth in raw_depths:
            try:
                depth_im = self.cv_bridge.imgmsg_to_cv2(
                    raw_depth, desired_encoding="passthrough")
                if np.max(depth_im) > 100:
                    depth_ims.append(depth_im / 1000.0)  # convert from [mm] to [m]
                else:
                    depth_ims.append(depth_im)
            except:
                rospy.logerr("Could not convert ROS depth image.")
        depth_ims = np.array(depth_ims)
        depth_ims[depth_ims == 0.0] = np.nan
        depth_image = np.nanmean(depth_ims, axis=0)
        depth_image[np.isnan(depth_image)] = 0.0

        return depth_image, raw_camera_info.K, raw_camera_info.header.frame_id

    def plan_grasp(self, req):
        """Grasp planner request handler.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS `ServiceRequest` for grasp planner service.
        """
        depth_im, K, frame = self.read_images(req)
        rospy.loginfo("Planning Grasp")
        return self._find_grasps(depth_im, self.gpnetplus, K, frame)

    def _find_grasps(self, depth_im, net, K, grasp_frame):
        """Executes a grasping policy on an `RgbdImageState`.

        Parameters
        ----------
        depth_im: :obj:`RgbdImageState`
            `RgbdImageState` from BerkeleyAutomation/perception to encapsulate
            depth and color image along with camera intrinsics.
        net: :obj:`GraspingPolicy`
            Grasping policy to use.
        grasp_frame: :obj:`str`
            Frame of reference to publish pose in.
        """
        # Execute the policy's action.
        rospy.loginfo("Grasp Planning Node: Predict grasps")
        grasps, scores, coordinates, toc = net(depth_im, K)

        # Create `GraspProposals` return msg and populate it.
        grasp_proposals = GraspProposals()
        grasp_proposals.header.frame_id = grasp_frame
        grasp_proposals.header.stamp = rospy.Time.now()
        for grasp, quality, coords in zip(grasps, scores, coordinates):
            gpnet_grasp = GPnetGrasp()
            gpnet_grasp.quality = quality
            pose = grasp.pose.squeeze()
            gpnet_grasp.pose.position.x = pose[0, 3]
            gpnet_grasp.pose.position.y = pose[1, 3]
            gpnet_grasp.pose.position.z = pose[2, 3]
            q = matrix_to_quaternion(pose[:3, :3])
            gpnet_grasp.pose.orientation.x = q[0]
            gpnet_grasp.pose.orientation.y = q[1]
            gpnet_grasp.pose.orientation.z = q[2]
            gpnet_grasp.pose.orientation.w = q[3]
            gpnet_grasp.width = grasp.width
            gpnet_grasp.image_u = coords[0]
            gpnet_grasp.image_v = coords[1]
            grasp_proposals.grasps.append(gpnet_grasp)
        return grasp_proposals


if __name__ == "__main__":
    # Initialize the ROS node.
    rospy.init_node("Grasp_Proposal_Server")

    # Initialize `CvBridge`.
    cv_bridge = CvBridge()

    # Get configs.
    model_name = rospy.get_param("~model_name")
    model_dir = rospy.get_param("~model_dir")
    if model_dir.lower() == "default":
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "../models")
    model_dir = os.path.join(model_dir, model_name)

    # Create a grasping policy.
    rospy.loginfo("Load Grasp Proposal Network GPnetPlus")
    grasp_proposal_network = GPnetPlus(model_dir)

    # Create a grasp planner.
    grasp_planner = GraspPlanner(cv_bridge, grasp_proposal_network)

    # Initialize the ROS service.
    grasp_planning_service = rospy.Service("gpnetplus_grasp_planner", GPnetGraspPlanner,
                                           grasp_planner.plan_grasp)
    rospy.loginfo("Grasp Planner Initialized")

    # Spin forever.
    rospy.spin()
