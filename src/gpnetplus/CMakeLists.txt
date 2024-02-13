cmake_minimum_required(VERSION 2.8.3)
project(gpnetplus)

find_package(catkin REQUIRED COMPONENTS
  genmsg
  geometry_msgs
  genmsg
  cv_bridge
  rospy
  std_msgs
  image_transport
  message_generation
  sensor_msgs
)

 ## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   GPnetPlusGrasp.msg
   GraspProposals.msg
   BoundingBox.msg
   BoundingBoxes.msg
 )

## Generate services in the 'srv' folder
 add_service_files(
  FILES
  GPnetPlusGraspPlanner.srv
  ObjectDetector.srv
 )


 ## Generate added messages with the following dependencies:
generate_messages(
  DEPENDENCIES
  geometry_msgs std_msgs sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime)

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/grasp_planning_node.py
  scripts/depth_im_test.py
  scripts/object_detection_node.py
  scripts/object_detection_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})