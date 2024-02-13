# Proposing Grasps for Mobile Manipulators in Domestic Environments

This is a ROS package for GP-net+ to be used on mobile manipulators. It uses a 
GP-net+ model to propose grasps on objects shown in depth images. A pre-trained model
for a robot with a PAL parallel jaw gripper is available at [zenodo](https://zenodo.org/records/10653956).
If you want to use GP-net+ for alternative grippers, you have to train a new model by
generating a new training dataset and train a new model with our code available on [GitHub](https://github.com/AuCoRoboticsMU/GP-netplus).


-----
### Installation

This code has been tested with a ROS noetic installation and python 3.8.10.
Besides the ROS installation, the package requirements are listed in `requirements.txt`.
We recommend an installation via docker or a virtual environment.

After cloning the `gpnetplus` package contained in `/src` into the `/src` folder of your catkin workspace, build 
the workspace and `source catkin_ws/devel/setup.bash` in order to use the package.

---
### Use GP-net+ to grasp objects

GP-net+ can be used by launching the `grasp_planning_node.py` and use the planning service by
sending a depth image and the camera info to the node, e.g.:

```
plan_grasp = rospy.ServiceProxy('gpnetplus_grasp_planner', GPnetPlusGraspPlanner)
depth_im = rospy.wait_for_message('/camera/depth_image', Image)
camera_intr = rospy.wait_for_message('/camera/info', CameraInfo)
grasp_response = plan_grasp(depth_im, camera_intr) 
```

An example usage script is given in `scripts/tiago_example.py`, which can be used with

`roslaunch gpnetplus tiago_example.launch`

This script also show-cases how an object detector model can be used to achieve target-driven grasping of objects.
It uses a pre-trained FASTER-RCNN model from Pytorch to allocate object ID's to grasp proposals, which can then be
selected in a GUI build with TKinter.

Note that you will have to adjust the `model_dir` in the launch file to the path were you
your GP-net+ model is stored. A pretrained model is available at [zenodo](https://zenodo.org/record/10653956).

----------------
If you use this code, please cite

A. Konrad, J. McDonald and R. Villing, "Learning to Grasp Unknown Objects in Domestic Environments," in review.

--------
### Acknowledgements

This publication has emanated from research supported in part by Grants from Science Foundation Ireland under 
Grant numbers 18/CRT/6049 and 16/RI/3399.
The opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do 
not necessarily reflect the views of the Science Foundation Ireland.