#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np

from object_detection_model import FasterRCNN

from tiago_object_detection.srv import ObjectDetector
from tiago_object_detection.msg import BoundingBox, BoundingBoxes

class ObjectDetection(object):
    def __init__(self, cv_bridge):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_bridge = cv_bridge
        self.fasterrcnn = FasterRCNN().to(device)

    def read_images(self, req):
        """Reads images from a ROS service request.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS ServiceRequest for object detection service.
        """
        # Get the raw depth image as ROS `Image` objects.
        raw_rgb = req.rgb_image

        rgb_image = self.cv_bridge.imgmsg_to_cv2(
            raw_rgb, desired_encoding="passthrough")

        return np.array(rgb_image)

    def run_object_detection(self, req):
        rgb_im = self.read_images(req)
        timestamp = req.rgb_image.header.stamp
        return self._get_bounding_boxes(rgb_im, req.rgb_image.header.frame_id, timestamp)

    def _get_bounding_boxes(self, depth_im, grasp_frame, timestamp):

        # Execute the policy's action.
        rospy.loginfo("Object Detection Node: Predict bounding boxes")

        boxes, scores, labels = self.fasterrcnn(depth_im, timestamp)

        # Create `GraspProposals` return msg and populate it.
        bounding_boxes = BoundingBoxes()
        bounding_boxes.header.frame_id = grasp_frame
        bounding_boxes.header.stamp = rospy.Time.now()
        cnt = 0
        for box, score, label in zip(boxes, scores, labels):
            if score < 0.6:
                continue
            bounding_box = BoundingBox()
            bounding_box.xmin = int(round(box[0]))
            bounding_box.ymin = int(round(box[1]))
            bounding_box.xmax = int(round(box[2]))
            bounding_box.ymax = int(round(box[3]))
            bounding_box.area = int(round((box[2] - box[0]) * (box[3] - box[1])))
            bounding_box.probability = score
            bounding_box.id = cnt
            bounding_box.Class = label
            bounding_boxes.bounding_boxes.append(bounding_box)

            cnt += 1
        return bounding_boxes


if __name__ == "__main__":
    # Initialize the ROS node.
    rospy.init_node("Object_Detection_Server")

    # Initialize `CvBridge`.
    cv_bridge = CvBridge()

    # Create a grasp planner.
    grasp_planner = ObjectDetection(cv_bridge)

    # Initialize the ROS service.
    grasp_planning_service = rospy.Service("object_detector", ObjectDetector,
                                           grasp_planner.run_object_detection)
    rospy.loginfo("Object Detector Initialized")

    # Spin forever.
    rospy.spin()
