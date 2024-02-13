#!/usr/bin/env python3
import copy
import sys
import rospy
import numpy as np
import math
import tkinter as tk

from transform import matrix_to_quaternion, quaternion_rotation_matrix

from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Quaternion, Point, TransformStamped
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from tf2_geometry_msgs import do_transform_pose
from moveit_msgs.msg import DisplayTrajectory
import tf2_ros

from moveit_msgs.srv import GetPlanningScene
from std_srvs.srv import Empty, EmptyRequest

from actionlib import SimpleActionClient

from gpnetplus.srv import GPnetPlusGraspPlanner
import moveit_commander
from moveit_commander import PlanningSceneInterface, MoveGroupCommander
from gpnetplus.srv import ObjectDetector


class GraspService(object):
    def __init__(self):
        rospy.loginfo("Starting Grasp Service")
        self.grasp_type = GraspObject()
        rospy.loginfo("Finished Grasp Service constructor")
        self.pick_gui = rospy.Service("/grasp_obj", Empty, self.start_grasp_obj)

    def start_grasp_obj(self, req):
        self.grasp_type.grasp_object()
        return {}


class GraspObject(object):
    def __init__(self):
        rospy.loginfo("Initalizing...")
        self.gs = GraspServer()

        rospy.loginfo("Initiate MoveGroupCommander")
        self.group = MoveGroupCommander('arm_torso')

        self.robot = moveit_commander.RobotCommander()
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            DisplayTrajectory,
                                                            queue_size=20)

        rospy.loginfo("Setting publishers to torso and head controller...")
        self.head_cmd = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=1)
        self.gripper_cmd = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=1)

        self.scene = PlanningSceneInterface()
        rospy.loginfo("Connecting to /get_planning_scene service")
        self.scene_srv = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
        self.scene_srv.wait_for_service()

        rospy.loginfo("Connecting to clear octomap service...")
        self.clear_octomap_srv = rospy.ServiceProxy('/clear_octomap', Empty)
        self.clear_octomap_srv.wait_for_service()

        rospy.loginfo("Waiting for '/play_motion' AS...")
        self.play_m_as = SimpleActionClient('/play_motion', PlayMotionAction)
        if not self.play_m_as.wait_for_server(rospy.Duration(20)):
            rospy.logerr("Could not connect to /play_motion AS")
            exit()

        rospy.loginfo("Waiting for '/parallel_gripper_controller/grasp' AS...")
        rospy.wait_for_service('/parallel_gripper_controller/grasp')
        self.gripper_controller = rospy.ServiceProxy('parallel_gripper_controller/grasp', Empty)

        self.furniture = 'Shelf'

        rospy.loginfo("Grasp client initialised.")

    def grasp_object(self):
        # Move TIAGo to pregrasp_position
        head_tilt = -0.8
        self.move_head(tilt=head_tilt)
        self.move_to_pregrasp(height=0.3)

        # Clear octomap
        self.clear_octomap_srv.call(EmptyRequest())

        input("Plan grasps.")
        possible_grasps, possible_grasp_poses = self.gs.predict_grasps()

        rospy.loginfo("Received {} grasp proposals.".format(len(possible_grasps)))
        success = False
        i = 0
        self.check_for_collisions_octomap()
        while not success and i < len(possible_grasps):
            grasp_pose = possible_grasps[i][0]
            grasp_width = possible_grasps[i][2]

            rospy.loginfo("Attempting grasp #{}".format(i))
            i += 1
            if grasp_width > 0.08:
                rospy.loginfo("Grasp #{} width exceeds gripper dimensions with {}".format(i, grasp_width))
                continue

            success = self.execute_grasp(grasp_pose)

        # Open gripper
        self.open_gripper()

    def execute_grasp(self, grasp_pose):
        grasp_pose.orientation = self.gs.normalize_quaternion(grasp_pose.orientation)
        self.gs.publish_poses([grasp_pose,
                               self.gs.calculate_pre_grasp_pose(grasp_pose,
                                                                dist=0.07,
                                                                end_effector=False)],
                              'base_footprint')

        end_effector_pose = self.gs.transform_tcp_to_end_effector(grasp_pose)
        pre_grasp_pose = self.gs.calculate_pre_grasp_pose(end_effector_pose, dist=0.07)

        self.group.clear_pose_targets()
        self.group.set_pose_target(pre_grasp_pose)
        plan_success, plan, planning_time, error_code = self.group.plan()
        if plan.joint_trajectory.points:
            self.display_trajectory(plan)
            resp = input("Can reach pose. continue? y/n: ")
            if resp != 'y':
                rospy.loginfo("User opted to not go to grasp pose. Check next grasp pose")
                return False

            self.group.execute(plan)
            self.group.clear_pose_targets()
        else:
            rospy.loginfo("Could not plan path to approach position. Abort.")
            return False

        pre_grasp_pose_2 = self.gs.calculate_pre_grasp_pose(end_effector_pose, dist=0.04)

        (grasp_plan, fraction) = self.group.compute_cartesian_path([pre_grasp_pose_2, end_effector_pose],
                                                                   0.01,
                                                                   0.0,
                                                                   avoid_collisions=False)
        self.group.execute(grasp_plan)

        # Close gripper
        self.gripper_controller(EmptyRequest())

        # Move up (change z coordinate of pose)
        post_grasp_pose = copy.deepcopy(end_effector_pose)
        post_grasp_pose.position.z += 0.05

        (grasp_plan, fraction) = self.group.compute_cartesian_path([post_grasp_pose],
                                                                   0.01,
                                                                   0.0,
                                                                   avoid_collisions=False)
        self.group.execute(grasp_plan)
        return True

    def display_trajectory(self, plan):
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory)

    def move_to_pregrasp(self, height=0.15):
        rospy.loginfo("Resetting TIAGo's pose to initial pose")

        joint_goal = self.group.get_current_joint_values()
        joint_goal[0] = height
        joint_goal[1] = 0.2
        joint_goal[2] = -0.07
        joint_goal[3] = -3.0
        joint_goal[4] = 1.5
        joint_goal[5] = -1.57
        joint_goal[6] = 0.2
        joint_goal[7] = 0.0
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(joint_goal)
        plan_success, plan, planning_time, error_code = self.group.plan()
        self.display_trajectory(plan)
        output = input('move')
        if output == 'n':
            return
        self.group.execute(plan, wait=True)
        self.group.stop()

    def open_gripper(self):
        jt = JointTrajectory()
        jt.joint_names = ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [0.04, 0.04]
        pt.time_from_start = rospy.Duration(0.5)
        jt.points.append(pt)
        self.gripper_cmd.publish(jt)
        rospy.sleep(1.0)

    def check_for_collisions_octomap(self):
        """ Let Tiago move his head to check the full perimeter for objects.
        """
        jt = JointTrajectory()
        cnt1 = 0.0
        cnt2 = 0.0
        jt.joint_names = ['head_1_joint', 'head_2_joint']
        if self.furniture == 'Table':
            tilt = -0.8
            for cnt2, turn in enumerate([0.0, 0.6, 0.0, -0.6, 0.0]):
                jtp = JointTrajectoryPoint()
                jtp.positions = [turn, tilt]
                jtp.time_from_start = rospy.Duration((float(cnt2)) * 3.0)
                jt.points.append(jtp)
        else:
            head_tilt = [0.5, -0.1, -0.7]
            for cnt1, tilt in enumerate(head_tilt):
                for cnt2, turn in enumerate([0.6, 0.0, -0.6, 0.0]):
                    jtp = JointTrajectoryPoint()
                    jtp.positions = [turn, tilt]
                    jtp.time_from_start = rospy.Duration((cnt1 * 4.0 + float(cnt2)) * 3.0)
                    jt.points.append(jtp)
        # Will not wait until command is finished!
        self.head_cmd.publish(jt)
        rospy.sleep((cnt1 * 4.0 + float(cnt2)) * 3.0 + 1.0)

    def move_head(self, tilt=0.0, turn=0.0):
        """ Let Tiago move his head.

            Parameters
            -------
            tilt (float): New rotational angle for the head tilting joint.
                            Negative is looking down, positive looking up.
            turn (float): New rotational angle for the head turning joint.

        """
        rospy.loginfo("GraspServer: Moving head")
        jt = JointTrajectory()
        jt.joint_names = ['head_1_joint', 'head_2_joint']
        jtp = JointTrajectoryPoint()
        jtp.positions = [turn, tilt]
        jtp.time_from_start = rospy.Duration(1.0)
        jt.points.append(jtp)
        # Will not wait until command is finished!
        self.head_cmd.publish(jt)


class GraspServer(object):
    def __init__(self):
        rospy.loginfo("Initializing GraspServer...")

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_l = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_b = tf2_ros.TransformBroadcaster()

        self.poses_pub = rospy.Publisher('/grasp_poses', PoseArray, latch=True, queue_size=3)
        self.object_detection = rospy.get_param("~object_detection")

        if self.object_detection:
            rospy.wait_for_service('object_detector')
            self.find_objects = rospy.ServiceProxy('object_detector', ObjectDetector)

        rospy.loginfo("GraspServer initialized!")

    def publish_poses(self, possible_poses, frame_id):
        pa = PoseArray()
        pa.header.frame_id = frame_id
        pa.header.stamp = rospy.Time.now()
        for pose in possible_poses:
            pa.poses.append(pose)
        self.poses_pub.publish(pa)

    def calculate_pre_grasp_pose(self, orig_pose, dist, end_effector=True):
        pose = copy.deepcopy(orig_pose)
        position = np.expand_dims(np.array((pose.position.x,
                                            pose.position.y,
                                            pose.position.z)),
                                  axis=-1)
        orientation = quaternion_rotation_matrix(pose.orientation)

        transform = np.vstack((np.hstack((orientation, position)),
                               np.array((0, 0, 0, 1))))
        if end_effector:
            translation = np.expand_dims(np.array((-dist, 0, 0, 1)), -1)
        else:
            translation = np.expand_dims(np.array((0, 0, -dist, 1)), -1)

        pre_grasp_position = np.matmul(transform, translation).squeeze()

        pre_grasp_pose = pose
        pre_grasp_pose.position.x = pre_grasp_position[0]
        pre_grasp_pose.position.y = pre_grasp_position[1]
        pre_grasp_pose.position.z = pre_grasp_position[2]

        return pre_grasp_pose

    def transform_tcp_to_end_effector(self, orig_pose):
        pose = copy.deepcopy(orig_pose)
        # Construct transform tcp
        position = np.expand_dims(np.array((pose.position.x,
                                            pose.position.y,
                                            pose.position.z)),
                                  axis=-1)
        orientation = quaternion_rotation_matrix(pose.orientation)

        tf_tcp = np.vstack((np.hstack((orientation, position)),
                            np.array((0, 0, 0, 1))))
        # Construct translation and rotation transform (rotate around y axis)
        tf = np.eye(4)
        tf[2, 3] = -0.22  # translation in z
        tf[0, 0] = 0
        tf[0, 2] = -1
        tf[2, 0] = 1
        tf[2, 2] = 0

        # Apply transform
        tf_end_effector = np.matmul(tf_tcp, tf).squeeze()

        # Generate new pose message
        end_effector_pose = pose
        end_effector_pose.position.x = tf_end_effector[0, 3]
        end_effector_pose.position.y = tf_end_effector[1, 3]
        end_effector_pose.position.z = tf_end_effector[2, 3]

        new_orientation = matrix_to_quaternion(tf_end_effector[:3, :3])

        end_effector_pose.orientation.x = new_orientation[0]
        end_effector_pose.orientation.y = new_orientation[1]
        end_effector_pose.orientation.z = new_orientation[2]
        end_effector_pose.orientation.w = new_orientation[3]

        return end_effector_pose

    def strip_leading_slash(self, s):
        return s[1:] if s.startswith("/") else s

    def normalize_quaternion(self, quat_msg, tolerance=0.0001):
        orig_quat = np.array((quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w))
        magnitude = sum(n * n for n in orig_quat)
        if magnitude > tolerance:
            magn_add = math.sqrt(magnitude)
            v = tuple(n / magn_add for n in orig_quat)
            normed_quat = Quaternion()
            normed_quat.x = v[0]
            normed_quat.y = v[1]
            normed_quat.z = v[2]
            normed_quat.w = v[3]
            return normed_quat
        return orig_quat

    def predict_grasps(self):
        rospy.wait_for_service('gpnetplus_grasp_planner')
        plan_grasp = rospy.ServiceProxy('gpnetplus_grasp_planner', GPnetGraspPlanner)

        # read images
        depth_images = []
        for i in range(10):
            depth_im = rospy.wait_for_message('/camera/depth_image', Image)
            depth_images.append(depth_im)
        rgb_img = rospy.wait_for_message('/camera/color_image', Image)
        camera_intr = rospy.wait_for_message('/camera/info', CameraInfo)
        response = plan_grasp(depth_images, camera_intr)
        rospy.loginfo("grasp_server: Received grasp proposals from GPnet+. Transform coordinate frames.")
        if self.object_detection:
            obj_detection_resp = self.find_objects(rgb_img)
            bounding_boxes = obj_detection_resp.bounding_boxes.bounding_boxes
        planned_grasps = response.grasp_proposals

        frame_id = self.strip_leading_slash(planned_grasps.header.frame_id)
        rospy.loginfo("grasp_server: Transforming from frame: " + frame_id + " to 'base_footprint'")
        grasps = []
        poses = []
        object_classes = []
        for idx, grasp in enumerate(planned_grasps.grasps):
            if grasp.width > 0.08:
                rospy.loginfo("Grasp too wide with {}m".format(grasp.width))
                continue
            ps = PoseStamped()
            ps.pose.position = grasp.pose.position
            ps.pose.orientation = grasp.pose.orientation
            object_class = 'None'

            ps.header.stamp = self.tfBuffer.get_latest_common_time("base_footprint", frame_id)
            ps.header.frame_id = frame_id

            transformed_ps = self.transform_to_base_footprint(ps, frame_id)

            if self.object_detection:
                grasp_image_u = grasp.image_u
                grasp_image_v = grasp.image_v
                minimal_area = 1000000

                for box in bounding_boxes:
                    # print("{}: {}".format(minimal_area, box.Class))
                    if box.xmin - 2 < grasp_image_v < box.xmax + 2 and \
                            box.ymin - 2 < grasp_image_u < box.ymax + 2 and box.area < minimal_area:
                        object_class = box.Class
                        minimal_area = box.area

            grasps.append([transformed_ps.pose, grasp.quality, grasp.width, object_class])
            object_classes.append(object_class)
            poses.append(transformed_ps.pose)
        self.publish_poses(poses, 'base_footprint')
        if self.object_detection:
            unique_object_classes = np.unique(np.array(object_classes))
            if len(unique_object_classes) > 1:
                self.chosen_object_num = None
                root_window = tk.Tk()
                root_window.title("Object-specific grasping")
                root_window.minsize(300, 80)
                label = tk.Label(root_window, text="Choose object to grasp:")
                label.pack()
                buttons = []
                for cnt, object_class in enumerate(unique_object_classes, start=0):
                    print(cnt)
                    buttons.append(tk.Button(root_window, text=object_class,
                                             command=lambda num=cnt: self.tkinter_button_clicked(num, root_window)))
                for button in buttons:
                    button.pack()
                root_window.wait_window()
                print(unique_object_classes[self.chosen_object_num])
                indices = np.where(np.array(object_classes) == unique_object_classes[self.chosen_object_num],
                                   True, False)
                grasps = np.array(grasps)[indices]
                poses = np.array(poses)[indices]
        return sorted(grasps, key=lambda l: - l[0].position.z), poses

    def tkinter_button_clicked(self, button_num, root_window):
        print(button_num)
        self.chosen_object_num = button_num
        root_window.destroy()

    def transform_to_base_footprint(self, ps, frame_id):
        transform_ok = False
        while not transform_ok and not rospy.is_shutdown():
            try:
                transform = self.tfBuffer.lookup_transform("base_footprint",
                                                           ps.header.frame_id,
                                                           rospy.Time(0))
                transformed_ps = do_transform_pose(ps, transform)
                transform_ok = True
            except tf2_ros.ExtrapolationException as e:
                rospy.logwarn("Exception on transforming point... trying again \n(" + str(e) + ")")
                rospy.sleep(0.01)
                ps.header.stamp = self.tfBuffer.get_latest_common_time("base_footprint", frame_id)
        return transformed_ps


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('grasping_demo')
    grasping = GraspService()
    rospy.spin()
