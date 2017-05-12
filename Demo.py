#!/usr/bin/env python
#encoding:utf-8
# Copyright (c) 2015, Fetch Robotics Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Fetch Robotics Inc. nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL FETCH ROBOTICS INC. BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Michael Ferguson

import copy
import actionlib
import rospy
from math import sin, cos
from moveit_python import (MoveGroupInterface,
                           PlanningSceneInterface,
                           PickPlaceInterface)
from moveit_python.geometry import rotate_pose_msg_by_euler_angles
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError, getCvType
import cv2
from control_msgs.msg import PointHeadAction, PointHeadGoal
from control_msgs.msg import (FollowJointTrajectoryAction,
                              FollowJointTrajectoryGoal,
                              GripperCommandAction,
                              GripperCommandGoal)
import numpy as np
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from moveit_msgs.msg import PlaceLocation, MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time
import random

import sys
from math import sin, cos, sqrt, pow, log10
import threading

from moveit_python import MoveGroupInterface
from geometry_msgs.msg import Pose, PoseStamped, Twist
from gazebo_msgs.msg import LinkStates, LinkState , ModelState , ModelStates
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header
from tf import TransformListener
import message_filters

import moveit_commander
import geometry_msgs.msg
import moveit_msgs.msg
import copy


IMAGE_SIZE = 84
PI = 3.14

# Move base using navigation stack
class MoveBaseClient(object):

    def __init__(self):
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base...")
        self.client.wait_for_server()

    def goto(self, x, y, theta, frame="map"):
        move_goal = MoveBaseGoal()
        move_goal.target_pose.pose.position.x = x
        move_goal.target_pose.pose.position.y = y
        # move_goal.target_pose.pose.orientation.z = sin(theta/2.0)
        move_goal.target_pose.pose.orientation.w = 1#cos(theta/2.0)
        move_goal.target_pose.header.frame_id = frame
        move_goal.target_pose.header.stamp = rospy.Time.now()

        # TODO wait for things to work
        self.client.send_goal(move_goal)
        self.client.wait_for_result()
    def go_reset(self,frame="map"):
        move_goal = MoveBaseGoal()
        move_goal.target_pose.pose.position.x = 0.0
        move_goal.target_pose.pose.position.y = 0.0
        # move_goal.target_pose.pose.position.z = 1.00137163018
        # move_goal.target_pose.pose.orientation.z = sin(theta/2.0)
        move_goal.target_pose.pose.orientation.w = 0.469066212873#cos(theta/2.0)
        # move_goal.target_pose.pose.orientation.x = 0.493365660624
        # move_goal.target_pose.pose.orientation.y = -0.635586111468
        # move_goal.target_pose.pose.orientation.z = 0.364139407051
        move_goal.target_pose.header.frame_id = frame
        move_goal.target_pose.header.stamp = rospy.Time.now()

        # TODO wait for things to work
        self.client.send_goal(move_goal)
        self.client.wait_for_result()
# Send a trajectory to controller
class FollowTrajectoryClient(object):

    def __init__(self, name, joint_names):
        self.client = actionlib.SimpleActionClient("%s/follow_joint_trajectory" % name,
                                                   FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for %s..." % name)
        self.client.wait_for_server()
        self.joint_names = joint_names

    def move_to(self, positions, duration=0.5):
        if len(self.joint_names) != len(positions):
            print("Invalid trajectory position")
            return False
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = positions
        trajectory.points[0].velocities = [0.0 for _ in positions]
        trajectory.points[0].accelerations = [0.0 for _ in positions]
        trajectory.points[0].time_from_start = rospy.Duration(duration)
        follow_goal = FollowJointTrajectoryGoal()
        follow_goal.trajectory = trajectory

        self.client.send_goal(follow_goal)
        self.client.wait_for_result()

# Point the head using controller
class PointHeadClient(object):

    def __init__(self):
        self.client = actionlib.SimpleActionClient("head_controller/point_head", PointHeadAction)
        rospy.loginfo("Waiting for head_controller...")
        self.client.wait_for_server()

    def look_at(self, x, y, z, frame, duration=1.0):
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z
        goal.min_duration = rospy.Duration(duration)
        self.client.send_goal(goal)
        self.client.wait_for_result()

class GripperClient(object):
    def __init__(self):
        self.client = actionlib.SimpleActionClient("gripper_controller/gripper_action", GripperCommandAction)
        rospy.loginfo("Waiting for gripper_controller...")
        self.client.wait_for_server()
        rospy.loginfo("...connected.")
    def action(self, position):
        gripper_goal = GripperCommandGoal()
        gripper_goal.command.max_effort = 10.0
        gripper_goal.command.position = position
        rospy.loginfo("Setting Gripper...")
        #print gripper_goal.command.position
        #if gripper_goal.command.position == 0.0:
        #    gripper_goal.command.position = 0.1 #0.0 close 0.1 open
        #elif gripper_goal.command.position == 0.1:
        #    gripper_goal.command.position = 0.0
        self.client.send_goal(gripper_goal)
        self.client.wait_for_result(rospy.Duration(2.0))
        rospy.loginfo("...done")

# Tools for grasping
class GraspingClient(object):

    def __init__(self):
        self.scene = PlanningSceneInterface("base_link")
        self.pickplace = PickPlaceInterface("arm", "gripper", verbose=True)
        self.move_group = MoveGroupInterface("arm", "base_link")

        find_topic = "basic_grasping_perception/find_objects"
        rospy.loginfo("Waiting for %s..." % find_topic)
        self.find_client = actionlib.SimpleActionClient(find_topic, FindGraspableObjectsAction)
        self.find_client.wait_for_server()

    def updateScene(self):
        # find objects
        goal = FindGraspableObjectsGoal()
        goal.plan_grasps = True
        self.find_client.send_goal(goal)
        self.find_client.wait_for_result(rospy.Duration(5.0))
        find_result = self.find_client.get_result()

        # remove previous objects
        for name in self.scene.getKnownCollisionObjects():
            self.scene.removeCollisionObject(name, False)
        for name in self.scene.getKnownAttachedObjects():
            self.scene.removeAttachedObject(name, False)
        self.scene.waitForSync()

        # insert objects to scene
        idx = -1
        for obj in find_result.objects:
            idx += 1
            obj.object.name = "object%d"%idx
            self.scene.addSolidPrimitive(obj.object.name,
                                         obj.object.primitives[0],
                                         obj.object.primitive_poses[0],
                                         wait = False)

        for obj in find_result.support_surfaces:
            # extend surface to floor, and make wider since we have narrow field of view
            height = obj.primitive_poses[0].position.z
            obj.primitives[0].dimensions = [obj.primitives[0].dimensions[0],
                                            1.5,  # wider
                                            obj.primitives[0].dimensions[2] + height]
            obj.primitive_poses[0].position.z += -height/2.0

            # add to scene
            self.scene.addSolidPrimitive(obj.name,
                                         obj.primitives[0],
                                         obj.primitive_poses[0],
                                         wait = False)

        self.scene.waitForSync()

        # store for grasping
        self.objects = find_result.objects
        self.surfaces = find_result.support_surfaces

    def getGraspableCube(self):
        graspable = None
        for obj in self.objects:
            # need grasps
            if len(obj.grasps) < 1:
                continue
            # check size
            if obj.object.primitives[0].dimensions[0] < 0.05 or \
               obj.object.primitives[0].dimensions[0] > 0.07 or \
               obj.object.primitives[0].dimensions[0] < 0.05 or \
               obj.object.primitives[0].dimensions[0] > 0.07 or \
               obj.object.primitives[0].dimensions[0] < 0.05 or \
               obj.object.primitives[0].dimensions[0] > 0.07:
                continue
            # has to be on table
            if obj.object.primitive_poses[0].position.z < 0.5:
                continue
            return obj.object, obj.grasps
        # nothing detected
        return None, None

    def getSupportSurface(self, name):
        for surface in self.support_surfaces:
            if surface.name == name:
                return surface
        return None

    def getPlaceLocation(self):
        pass

    def pick(self, block, grasps):
        success, pick_result = self.pickplace.pick_with_retry(block.name,
                                                              grasps,
                                                              support_name=block.support_surface,
                                                              scene=self.scene)
        self.pick_result = pick_result
        return success

    def place(self, block, pose_stamped):
        places = list()
        l = PlaceLocation()
        l.place_pose.pose = pose_stamped.pose
        l.place_pose.header.frame_id = pose_stamped.header.frame_id

        # copy the posture, approach and retreat from the grasp used
        l.post_place_posture = self.pick_result.grasp.pre_grasp_posture
        l.pre_place_approach = self.pick_result.grasp.pre_grasp_approach
        l.post_place_retreat = self.pick_result.grasp.post_grasp_retreat
        places.append(copy.deepcopy(l))
        # create another several places, rotate each by 360/m degrees in yaw direction
        m = 16 # number of possible place poses
        pi = 3.141592653589
        for i in range(0, m-1):
            l.place_pose.pose = rotate_pose_msg_by_euler_angles(l.place_pose.pose, 0, 0, 2 * pi / m)
            places.append(copy.deepcopy(l))

        success, place_result = self.pickplace.place_with_retry(block.name,
                                                                places,
                                                                scene=self.scene)
        return success

    def tuck(self):
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        pose = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(joints, pose, 0.02)
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                return
    def mytuck(self):
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        # pose = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        pose = [random.uniform(-0.5,2.0) for i in range(7)]
        print pose
        self.move_group.moveToJointPosition(joints, pose, 0.02)
    def my_arm_control(self,t_angle):
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        # pose = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        print t_angle
        self.move_group.moveToJointPosition(joints, t_angle, 0.02)
import os
velocities = np.zeros((7),np.float32)
def callback(image):

    velocities[0] = 0.01*(cv2.getTrackbarPos('1', window_name) - 314)
    velocities[1] = 0.01*(cv2.getTrackbarPos('2', window_name) - 314)
    velocities[2] = 0.01*(cv2.getTrackbarPos('3', window_name) - 314)
    velocities[3] = 0.01*(cv2.getTrackbarPos('4', window_name) - 314)
    velocities[4] = 0.01*(cv2.getTrackbarPos('5', window_name) - 314)
    velocities[5] = 0.01*(cv2.getTrackbarPos('6', window_name) - 314)
    velocities[6] = 0.01*(cv2.getTrackbarPos('7', window_name) - 314)
    bridge_ = CvBridge()
    cvimg = bridge_.imgmsg_to_cv2(image, "bgr8")
    cv2.imshow(window_name,cvimg)
    # group_variable_values = arm_group.get_current_joint_values()
    # print "============ Joint values: ", group_variable_values
    cv2.waitKey(1)

def do_nothing(abc):
    return

trackbar_size = 628
window_name = 'arm angle'
img = np.zeros((300,400,3),np.uint8)
min_size = 0
max_size = 628
cv2.namedWindow(window_name)
cv2.createTrackbar('1', window_name, min_size,max_size, do_nothing)
cv2.createTrackbar('2', window_name, min_size,max_size, do_nothing)
cv2.createTrackbar('3', window_name, min_size,max_size, do_nothing)
cv2.createTrackbar('4', window_name, min_size,max_size, do_nothing)
cv2.createTrackbar('5', window_name, min_size,max_size, do_nothing)
cv2.createTrackbar('6', window_name, min_size,max_size, do_nothing)
cv2.createTrackbar('7', window_name, min_size,max_size, do_nothing)

def set_arm_angle(velocities):

    velocities[0] = 0.01*(cv2.getTrackbarPos('1', window_name) - 314)
    velocities[1] = 0.01*(cv2.getTrackbarPos('2', window_name) - 314)
    velocities[2] = 0.01*(cv2.getTrackbarPos('3', window_name) - 314)
    velocities[3] = 0.01*(cv2.getTrackbarPos('4', window_name) - 314)
    velocities[4] = 0.01*(cv2.getTrackbarPos('5', window_name) - 314)
    velocities[5] = 0.01*(cv2.getTrackbarPos('6', window_name) - 314)
    velocities[6] = 0.01*(cv2.getTrackbarPos('7', window_name) - 314)
    cv2.imshow(window_name,img)
    cv2.waitKey(1)
    return velocities





def move_left(move_base):
    move_base.goto(-1.,0.,0.)

def move_right(move_base):
    move_base.goto(1.,0.,0.)

def move_up(move_base):
    move_base.goto(1.,1.,0)

def move_down(move_base):
    move_base.goto(0.,-1.,0)

def move_free(move_base):
    x = float(raw_input("x : "))
    y = float(raw_input("y : "))
    a = float(raw_input("alpha : "))
    move_base.goto(x,y,a)




waypoints = []

def move_reset(group):
    # start with the current pose
    # first orient gripper and move forward (+x)
    xx = float(raw_input('x:'))
    yy = float(raw_input('y:'))
    zz = float(raw_input('z:'))
    wpose = geometry_msgs.msg.Pose()
    wpose.orientation.w = 1.0
    wpose.position.x = waypoints[0].position.x + xx#+ 0.1
    wpose.position.y = waypoints[0].position.y + yy 
    wpose.position.z = waypoints[0].position.z + zz
    waypoints.append(copy.deepcopy(wpose))

    # second move down
    wpose.position.z -= 0.10
    waypoints.append(copy.deepcopy(wpose))

    # third move to the side
    wpose.position.y += 0.05
    waypoints.append(copy.deepcopy(wpose))

    (plan3, fraction) = group.compute_cartesian_path(
                             waypoints,   # waypoints to follow
                             0.01,        # eef_step
                             0.0)         # jump_threshold
    group.go(wait = True)
def control_gripper(gripper_goal):
    rospy.loginfo("Setting positions...")
    print gripper_goal.command.position

    if gripper_goal.command.position == 0.0:
        gripper_goal.command.position = 0.1 #0.0 close 0.1 open
    elif gripper_goal.command.position == 0.1:
        gripper_goal.command.position = 0.0

    gripper_client.send_goal(gripper_goal)
    gripper_client.wait_for_result(rospy.Duration(2.0))
    rospy.loginfo("...done")
def open_gripper(gripper_goal):
    rospy.loginfo("Setting positions...")
    print gripper_goal.command.position
    gripper_goal.command.position = 0.1 #0.0 close 0.1 open
    gripper_client.send_goal(gripper_goal)
    gripper_client.wait_for_result(rospy.Duration(2.0))
    rospy.loginfo("...done")
def close_gripper(gripper_goal):
    rospy.loginfo("Setting positions...")
    print gripper_goal.command.position
    gripper_goal.command.position = 0.0 #0.0 close 0.1 open
    gripper_client.send_goal(gripper_goal)
    gripper_client.wait_for_result(rospy.Duration(2.0))
    rospy.loginfo("...done")

class robotGame():
    def __init__(self):
        #init code
        rospy.init_node("demo")
        while not rospy.Time.now():
            pass
        self.currentDist = 1
        self.previousDist = 1
        self.reached = False
        self.tf = TransformListener()
        self.state_lock = threading.Lock()
        self.joints_lock = threading.Lock()
        self.gripper_pose_lock = threading.Lock()
        self.cube_pose_lock = threading.Lock()
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        self.joint_positions = [0, 0, 0, 0, 0, 0, 0]
        self.jv = []
        self.gripper_position = 0.1
        self.gp = []
        self.move_base = MoveBaseClient()
        self.head_action = PointHeadClient()
        self.arm_action = MoveGroupInterface("arm", "base_link")
        self.arm_controller = FollowTrajectoryClient("arm_controller", self.joint_names)
        self.torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
        self.gripper_client = GripperClient()
        self.cubes_index = []
        self.cubes_pose = LinkStates()

        self.body_pose = ModelState() # 身体坐标角度等参数
        self.body_pose.model_name = 'fetch' #身体
        self.body_pose.pose = Pose()        #各个
        self.body_pose.twist = Twist()      #参数
        self.body_pose.reference_frame = 'map'  #初始化

        self.grippers_index = []
        self.grippers_pose = LinkStates()
        self.states_sub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.getRealtimePose)
        self.cube_pose_pub = rospy.Publisher('/gazebo/set_link_state', LinkState, queue_size=1)
        self.body_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        self.cp = LinkState()
        self.cp.link_name = 'demo_cube::link'
        self.cp.pose = Pose()
        self.cp.twist = Twist()
        self.cp.reference_frame = 'map'
        # Get realtime image
        self.image_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image)
        self.ts = message_filters.TimeSynchronizer([self.image_sub], 10)
        self.ts.registerCallback(self.getRealtimeImage)
        self.js = JointState()
        self.js.header = Header()
        self.js.name = self.joint_names
        self.js.velocity = []
        self.js.effort = []
        self.sub = rospy.Subscriber('/joint_states', JointState, self.jsCB)
        self.destPos = np.random.uniform(0,0.25, size =(3))

        rospy.loginfo("Raising torso...")
        self.torso_action.move_to([0.4, ])

    def getRealtimePose(self, data):
        try:
            self.cube_pose_lock.acquire()
            if len(self.cubes_index) == 0:
                for idx, link in enumerate(data.name):
                    if link.startswith('demo_cube::'):
                        self.cubes_index.append(idx)
                        cubes_name = 'demo_cube_' + link[11:]
                        self.cubes_pose.name.append(cubes_name)
                self.cubes_pose.pose = [Pose()] * len(self.cubes_index)
            for i, c_index in enumerate(self.cubes_index):
                self.cubes_pose.pose[i] = data.pose[c_index]
        except ValueError, e:
            print "[ValueError] %s for get realtime cubes pose" % e
        finally:
            self.cube_pose_lock.release()
        try:
            self.gripper_pose_lock.acquire()
            flag = 0
            if len(self.grippers_index) == 0:
                for idx, link in enumerate(data.name):
                    if link.startswith('fetch::l_gripper'):
                        self.grippers_index.append(idx)
                        flag += 1
                    if link.startswith('fetch::r_gripper'):
                        self.grippers_index.append(idx)
                        flag += 1
                    if flag >= 2:
                        break
                self.grippers_pose.pose = [Pose()] * 1
                self.gp = [0.0] * 1
            self.grippers_pose.name.append('fetch_gripper')
            self.grippers_pose.pose[0].position.x = (data.pose[self.grippers_index[0]].position.x + data.pose[self.grippers_index[1]].position.x) / 2
            self.grippers_pose.pose[0].position.y = (data.pose[self.grippers_index[0]].position.y + data.pose[self.grippers_index[1]].position.y) / 2
            self.grippers_pose.pose[0].position.z = (data.pose[self.grippers_index[0]].position.z + data.pose[self.grippers_index[1]].position.z) / 2
            self.gp[0] = sqrt(pow(data.pose[self.grippers_index[0]].position.x - data.pose[self.grippers_index[1]].position.x, 2) + \
            pow(data.pose[self.grippers_index[0]].position.y - data.pose[self.grippers_index[1]].position.y, 2) + \
            pow(data.pose[self.grippers_index[0]].position.z - data.pose[self.grippers_index[1]].position.z, 2))
        except ValueError, e:
            print "[ValueError] %s for get realtime gripper pose" % e
        finally:
            self.gripper_pose_lock.release()
    def jsCB(self,msg):
        temp_dict = dict(zip(msg.name, msg.position))
        self.joints_lock.acquire()
        self.jv = [temp_dict[x] for x in self.joint_names]
        self.js.position = self.jv
        self.joints_lock.release()
        self.js.header.stamp = rospy.Time.now()
    def getRealtimeImage(self, image):
        self.state_lock.acquire()
        bridge_ = CvBridge()
        self.cvimg = cv2.resize(bridge_.imgmsg_to_cv2(image, "bgr8"), (IMAGE_SIZE, IMAGE_SIZE))
        self.state_lock.release()
        #cv2.imshow('head_camera',cvimg)
        #cv2.waitKey(1)
    def getCurrentState(self):
        self.state_lock.acquire()
        cvimg = self.cvimg
        self.state_lock.release()
        return cvimg
    def getCurrentJointValues(self):
        self.joints_lock.acquire()
        self.gripper_pose_lock.acquire()
        jv = self.jv
        gp = self.gp[0]
        self.joints_lock.release()
        self.gripper_pose_lock.release()
        return [jv, gp]

    def getCurrentGripperPose(self):
        self.gripper_pose_lock.acquire()
        gripper_pose_x = self.grippers_pose.pose[0].position.x
        gripper_pose_y = self.grippers_pose.pose[0].position.y
        gripper_pose_z = self.grippers_pose.pose[0].position.z
        self.gripper_pose_lock.release()
        return [gripper_pose_x, gripper_pose_y, gripper_pose_z]
    def getCurrentTargetPose(self):
        self.cube_pose_lock.acquire()
        cube_pose_x = self.cubes_pose.pose[0].position.x
        cube_pose_y = self.cubes_pose.pose[0].position.y
        cube_pose_z = self.cubes_pose.pose[0].position.z
        self.cube_pose_lock.release()
        return [cube_pose_x, cube_pose_y, cube_pose_z]
    def getState(self):
        self.joints_lock.acquire()
        self.gripper_pose_lock.acquire()
        self.cube_pose_lock.acquire()
        joint_pos=self.joint_positions
        gripper_pose_x = self.grippers_pose.pose[0].position.x
        gripper_pose_y = self.grippers_pose.pose[0].position.y
        gripper_pose_z = self.grippers_pose.pose[0].position.z
        cube_pose_x = self.cubes_pose.pose[0].position.x
        cube_pose_y = self.cubes_pose.pose[0].position.y
        cube_pose_z = self.cubes_pose.pose[0].position.z
        self.cube_pose_lock.release()
        self.gripper_pose_lock.release()
        self.joints_lock.release()
        return [joint_pos[0],joint_pos[1],joint_pos[2],joint_pos[3],joint_pos[4],joint_pos[5],joint_pos[6],gripper_pose_x,gripper_pose_y,gripper_pose_z,cube_pose_x,cube_pose_y,cube_pose_z]

    def getDist(self):
        position = self.getCurrentGripperPose()
        print "Current Gripper Position is %s" % position
        currentPos = np.array((position[0], position[1], position[2]))
        target_position = self.getCurrentTargetPose()
        print "Current Target Cube Position is %s" % target_position
        self.destPos = np.array((target_position[0], target_position[1], target_position[2]))
        print 'fuck ',position[0]-target_position[0],' , ',position[1]-target_position[1],' , ',position[2]-target_position[2]
        dist = np.linalg.norm(currentPos - self.destPos)
        print "Current dist is %s" % dist
        return dist
    def setJointValues(self, tjv, tjp):
        self.joint_positions = tjv
        self.gripper_position = tjp #close gripper
        rospy.sleep(0.20)
        return True
    def setCubePose(self):
        self.cp.pose.position.x = 0.7 + np.random.uniform(-0.15, -0.08)
        self.cp.pose.position.y = 0.0 + np.random.uniform(-0.2, 0.2)
        self.cp.pose.position.z = 0.8
        self.cp.pose.orientation.x = 0
        self.cp.pose.orientation.y = 0
        theta = np.random.uniform(0, 2 * PI)
        self.cp.pose.orientation.z = sin(theta/2)
        self.cp.pose.orientation.w = cos(theta/2)
        self.cube_pose_pub.publish(self.cp)
    def setBodyPose(self):
        #rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState "{model_name: fetch, pose: {position: {x: 0,y: 0,z: 0.0},orientation: {x: 0,y: 0,z: 0,w: 1.0}},twist: {}}"
        self.body_pose.pose.position.x = 0.0
        self.body_pose.pose.position.y = 0.0
        self.body_pose.pose.position.z = 0.0
        self.body_pose.pose.orientation.x = 0.0
        self.body_pose.pose.orientation.y = 0.0
        self.body_pose.pose.orientation.z = 0.0
        self.body_pose.pose.orientation.w = 1.0
        self.body_pub.publish(self.body_pose)
    def control_gripper(self):
        self.gripper_client.action(self.gripper_position)
    def control_joints(self):
        self.arm_controller.move_to(self.joint_positions)
    def change_arm_angle(self,angle_list):
        new_angle_list = [x + y for x,y in zip(angle_list, self.getCurrentJointValues()[0])]
        print 'new arm angle : ',new_angle_list
        self.joint_positions = new_angle_list
        self.control_joints()
    def getReward(self):
        curDist = self.getDist()
        print "Dist after current step is %s" % curDist
        reward = -curDist - 0.5*log10(curDist)
        if curDist < 0.01:
            reward +=10
            done = True
        print "reward is %s" % reward
        return reward
    def step(self, vals):
        done = False
        prevDist = self.getDist()
        print "Dist before current step is %s" % prevDist
        tvals = vals.flatten().tolist()
        print "Current action is %s" % tvals
        tjv = [x + y for x,y in zip(tvals[:-1], self.getCurrentJointValues()[0])]
        status = self.setJointValues(tvals[:-1], tvals[-1])
        self.control_joints()
        self.control_gripper()
        rospy.sleep(0.20)
        curDist = self.getDist()
        print "Dist after current step is %s" % curDist
        reward = -curDist - 0.00*np.linalg.norm(vals) - 0.5*log10(curDist)
        print self.destPos, -curDist - 0.5*log10(curDist), -curDist, np.linalg.norm(vals)
        if curDist < 0.01:
            reward +=10
            done = True
        print "reward is %s" % reward
        ts = self.getCurrentState()
        tjv, tgp = self.getCurrentJointValues()
        tjv.append(tgp)
        return [[ts, tjv], reward, done]
    def reset(self):

        print "reset body position..."
        self.setBodyPose()
        # rospy.sleep(0.10)
        self.setJointValues([1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0], 0.1)
        # self.setJointValues(np.random.uniform(-3.14, 3.14, size=(7)), 0.1)
        self.control_gripper()
        #self.arm_action.moveToJointPosition(self.joint_names, self.joint_positions, 0.02)
        print "Reset joint positions to %s" % self.joint_positions
        self.arm_controller.move_to(self.joint_positions)
        # rospy.sleep(0.10)
        self.destPos = np.random.uniform(0, 0.25, size =(3))
        print "Set cube new position..."
        self.setCubePose()
        # rospy.sleep(0.10)
        
        print "Cube new position is [%s, %s, %s]" % (self.cp.pose.position.x, self.cp.pose.position.y, self.cp.pose.position.z)
        print "Cube real new position is [%s, %s, %s]" % (self.cubes_pose.pose[0].position.x, self.cubes_pose.pose[0].position.y, self.cubes_pose.pose[0].position.z)
        # Point the head at the cube we want to pick
        rospy.loginfo("Looking at the cube...")
        self.head_action.look_at(self.cubes_pose.pose[0].position.x, self.cubes_pose.pose[0].position.y, self.cubes_pose.pose[0].position.z, "map")
        ts = self.getCurrentState()
        while len(self.jv) == 0 or len(self.gp) == 0:
            print "jv or gp is not ready"
            time.sleep(5)
        #print "current jv is %s" % self.jv
        #print "current gp is %s" % self.gp
        tjv, tgp = self.getCurrentJointValues()
        #print "tjv is %s, tgp is %s" % (tjv, tgp)
        tjv.append(tgp)
        return [ts, tjv]

    def done(self):
        self.sub.unregister()
        rospy.signal_shutdown("done")

    def auto_control_arm(self):
        dist = self.getDist()
        target_position = self.getCurrentTargetPose()

        while dist > 0.02:
            position = self.getCurrentGripperPose()
            currentPos = np.array((position[0], position[1], position[2]))
            
            dist = self.getDist()



import time

#-------------------------------------------------------------groups-------------------------------------------------
arm_group = moveit_commander.MoveGroupCommander("arm")
gripper_group = moveit_commander.MoveGroupCommander("gripper")
all_group = moveit_commander.MoveGroupCommander("arm_with_torso")
#-------------------------------------------------------------groups-------------------------------------------------

def out_to_string(a):
    si = ''
    for i in a[0]:
        si += str(i)+' '
    for i in a[1]:
        si += str(i)+' '
    si += str(a[2])+' '
    for i in a[3]:
        si += str(i)+' '
    si += str(a[-1])
    print si 
    return si
if __name__ == "__main__":
    # Create a node
    rospy.init_node("demo")
    waypoints.append(all_group.get_current_pose().pose)
    print waypoints
    print "============ Starting tutorial setup"
    moveit_commander.roscpp_initialize(sys.argv)
    # rospy.init_node('move_group_python_interface_tutorial',
                # anonymous=True)

    robot = moveit_commander.RobotCommander()
    print 'robot groups : ',robot.get_group_names() # robot groups :  ['arm', 'arm_with_torso', 'gripper']
    

    print "======= arm_group.get_planning_frame: %s" % arm_group.get_planning_frame()
    print "=======arm_group.get_end_effector_link(): %s" % arm_group.get_end_effector_link()

    print "======= gripper_group.get_planning_frame: %s" % gripper_group.get_planning_frame()
    print "=======gripper_group.get_end_effector_link(): %s" % gripper_group.get_end_effector_link()

    print "============ Printing robot state"
    print robot.get_current_state()
    print "============"

    

    # time.sleep(10)
    # Make sure sim time is working
    while not rospy.Time.now():
        pass

    rbs = robotGame()
    
    image_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image)
    # depth_img = message_filters.Subscriber('/head_camera/depth_downsample/image_raw', Image)
    # info_sub = message_filters.Subscriber('camera_info', CameraInfo)


    # Setup clients
    move_base = MoveBaseClient()
    torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    head_action = PointHeadClient()
    grasping_client = GraspingClient()

    rospy.loginfo("Waiting for gripper_controller...")
    gripper_client = actionlib.SimpleActionClient("gripper_controller/gripper_action", GripperCommandAction)
    gripper_client.wait_for_server()
    rospy.loginfo("...connected.")

    gripper_goal = GripperCommandGoal()
    gripper_goal.command.max_effort = 10.0
    gripper_goal.command.position = 0.1
    ts = message_filters.TimeSynchronizer([image_sub,], 10)
    ts.registerCallback(callback)

    t_angle = np.zeros((7),np.float32)
    xx = 0.0
    yy = 0.0
    zz = 0.0
    tx = 0.0
    ty = 0.0
    tz = 0.0
    mycmd = raw_input("your commond: ")
    while mycmd != 'q':

        if mycmd == 'a':
            move_left(move_base)
        elif mycmd == 'd':
            move_right(move_base)
        elif mycmd == 'w':
            move_up(move_base)
        elif mycmd == 's':
            move_down(move_base)
        elif mycmd == 'pos':
            print 'pose : ', all_group.get_current_pose().pose
        elif mycmd == 'm0':
            # move_reset(all_group)
            move_free(move_base)
        elif mycmd == 'm1':
            move_base.go_reset()
        elif mycmd == 'tt':
            grasping_client.tuck()
        elif mycmd == 'l': #body lower
            torso_action.move_to([0.0,])
        elif mycmd == 'r': #body raise
            torso_action.move_to([0.2,])
        elif mycmd == 't':
            grasping_client.mytuck()
        elif mycmd == 're':
            rbs.reset()
            target_position = rbs.getCurrentTargetPose()
            time.sleep(3)
            while target_position[2] < 0.7:
                print target_position[2]
                rbs.reset()
                time.sleep(3)
        elif mycmd == 'fuck':
            print "============ Generating plan 1"
            xx = float(raw_input('x:'))
            yy = float(raw_input('y:'))
            zz = float(raw_input('z:'))

            pose_target = geometry_msgs.msg.Pose()
            pose_target.orientation.w = 1.0
            pose_target.position.x = xx #0.7
            pose_target.position.y = yy #-0.05
            pose_target.position.z = zz #1.1
            arm_group.set_pose_target(pose_target)
            trajectory = arm_group.plan()
            print type(trajectory)
            position_list = [i.velocities for i in trajectory.joint_trajectory.points]
            print len(trajectory.joint_trajectory.points)
            # print '-------------------------------------------'
            # for i in trajectory.joint_trajectory.points:
            #     print '-------------------------------------------'
            #     print "positions",i.positions
            #     print "velocities",i.velocities
            #     print "accelerations",i.accelerations
            #     print '-------------------------------------------'
            # print '-------------------------------------------'
            # for i in position_list[:-1]:
            #     print 'move to ',i
                # rbs.change_arm_angle(i)
                # rbs.arm_controller.move_to(i)
                # rbs.getDist()
            
            # print type(trajectory.joint_trajectory.points)
            # print trajectory.joint_trajectory.points
            arm_group.go(wait = True)
            rbs.getDist()
            print xx,yy,zz
            position = rbs.getCurrentGripperPose()
            # print "Current Gripper Position is %s" % position
            currentPos = np.array((position[0], position[1], position[2]))
            target_position = rbs.getCurrentTargetPose()
            # print "Current Target Cube Position is %s" % target_position
            rbs.destPos = np.array((target_position[0], target_position[1], target_position[2]))
            tx = -target_position[0] + position[0]
            ty = -target_position[1] + position[1]
            tz = -target_position[2] + position[2]
            print 'follow to (a-b)',tx,ty,tz
            display_trajectory_publisher = rospy.Publisher(
                                            '/move_group/display_planned_path',
                                            moveit_msgs.msg.DisplayTrajectory)

            print "============ Visualizing plan1"
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = robot.get_current_state()
            display_trajectory.trajectory.append(trajectory)
            display_trajectory_publisher.publish(display_trajectory)
            group_variable_values = arm_group.get_current_joint_values()
            print "============ Joint values: ", group_variable_values
        elif mycmd == 'auto':
            fuck = 0
            shit = 0
            success_l = []
            while fuck < 100:

                #reset
                rbs.reset()
                target_position = rbs.getCurrentTargetPose()
                time.sleep(1)
                while target_position[2] < 0.5:
                    print target_position[2]
                    rbs.reset()
                    time.sleep(2)
                fuck += 1
                print 'stop reset'
                # open gripper 
                open_gripper(gripper_goal)
                print "============ Generating plan"
                target_position = rbs.getCurrentTargetPose()
                print "======================Current Target Cube Position is======================= %s" % target_position
                rbs.destPos = np.array((target_position[0], target_position[1], target_position[2]))
                tx = random.uniform(0.081,0.066)
                ty = random.uniform(-0.04,0.02)
                tz = random.uniform(-0.009,0.003)
                xx = target_position[0] - tx 
                yy = target_position[1] - ty 
                zz = target_position[2] - tz 
                print "======================================do action at=============================== ",xx,yy,zz
                
                pose_target = geometry_msgs.msg.Pose()
                pose_target.orientation.w = 1.0
                pose_target.position.x = xx - tx #0.7
                pose_target.position.y = yy - ty  #-0.05
                pose_target.position.z = zz - tz #1.1
                arm_group.set_pose_target(pose_target)
                trajectory = arm_group.plan()
                print type(trajectory)
                position_list = [i.velocities for i in trajectory.joint_trajectory.points]
                print len(trajectory.joint_trajectory.points)
                
                arm_group.go(wait = True)
                rbs.getDist()

                # print xx,yy,zz
                position = rbs.getCurrentGripperPose()
                # print "Current Gripper Position is %s" % position
                currentPos = np.array((position[0], position[1], position[2]))
                target_position = rbs.getCurrentTargetPose()
                # print "Current Target Cube Position is %s" % target_position
                rbs.destPos = np.array((target_position[0], target_position[1], target_position[2]))
                # tx = -target_position[0] + position[0]
                # ty = -target_position[1] + position[1]
                # tz = -target_position[2] + position[2]
                print u'-----------------------相差为---------------------',tx,ty,tz
                display_trajectory_publisher = rospy.Publisher(
                                                '/move_group/display_planned_path',
                                                moveit_msgs.msg.DisplayTrajectory)

                print "============ Visualizing plan1"
                display_trajectory = moveit_msgs.msg.DisplayTrajectory()
                display_trajectory.trajectory_start = robot.get_current_state()
                display_trajectory.trajectory.append(trajectory)
                display_trajectory_publisher.publish(display_trajectory)
                group_variable_values = arm_group.get_current_joint_values()
                print "============ Joint values: ", group_variable_values
                #close gripper
                close_gripper(gripper_goal)
                grasping_client.tuck()
                if_ac = rbs.getDist()
                print '~~~~~~~~~~~~~~~~~~~~距离~~~~~~~~~~~~~~~',if_ac
                if if_ac < 0.05:
                    print '!!!!!!!!!!!!!!!!!!!!success!!!!!!!!!!!!!!!!!!!!'
                    print "======================Current Target Cube Position is======================= %s" % target_position
                    print "======================================do action at=============================== ",xx,yy,zz
                    print u'-----------------------相差为---------------------',tx,ty,tz
                    success_l.append([tx,ty,tz])
                    # time.sleep(5)
                    shit += 1
                
                
                # time.sleep(1)
                print 'start gripping action times :',fuck
            print u'成功率:%d%%'%(100*(float(shit)/fuck))
            print u'成功列表为: '
            for i in success_l:
                print i
            ia = 0.
            ib = 0.
            ic = 0.
            for i in success_l:
                ia += i[0]
                ib += i[1]
                ic += i[2]
            print ia/len(success_l),ib/len(success_l),ic/len(success_l)
        elif mycmd == 'data':
            fuck = 0
            shit = 0
            success_l = []
            with open('data.txt','w') as f:

                while fuck < 10:
                    data_list = []
                    #reset
                    rbs.reset()
                    target_position = rbs.getCurrentTargetPose()
                    time.sleep(1)
                    while target_position[2] < 0.5:
                        print target_position[2]
                        rbs.reset()
                        time.sleep(2)
                    fuck += 1
                    print 'stop reset'
                    # open gripper 
                    open_gripper(gripper_goal)
                    print "============ Generating plan"
                    target_position = rbs.getCurrentTargetPose()
                    print "======================Current Target Cube Position is======================= %s" % target_position
                    rbs.destPos = np.array((target_position[0], target_position[1], target_position[2]))
                    tx = random.uniform(0.081,0.066)
                    ty = random.uniform(-0.04,0.02)
                    tz = random.uniform(-0.009,0.003)
                    xx = target_position[0] - tx 
                    yy = target_position[1] - ty 
                    zz = target_position[2] - tz 
                    print "======================================do action at=============================== ",xx,yy,zz
                    
                    pose_target = geometry_msgs.msg.Pose()
                    pose_target.orientation.w = 1.0
                    pose_target.position.x = xx - tx #0.7
                    pose_target.position.y = yy - ty  #-0.05
                    pose_target.position.z = zz - tz #1.1
                    arm_group.set_pose_target(pose_target)
                    trajectory = arm_group.plan()
                    print type(trajectory)
                    position_list = [i.positions for i in trajectory.joint_trajectory.points]
                    print len(trajectory.joint_trajectory.points)
                    
                    for i in range(len(position_list)-1):
                        # t_l 为state变量
                        t_l = np.zeros((14),np.float32)
                       
                        # 0-6 机械臂角度
                        for j in range(len(position_list[i])):
                            t_l[j] = position_list[i][j]
                        
                        # 爪子实时获取
                        # t_l[7] = 0.1
                        t_l[7] = rbs.gp[0]
                        # 得到爪子position
                        t_1 = rbs.getCurrentGripperPose()
                        # 设置爪子position
                        t_l[8] = t_1[0]
                        t_l[9] = t_1[1]
                        t_l[10] = t_1[2]
                        # 得到cube的position
                        t_2 = rbs.getCurrentTargetPose()
                        # 设置爪子position
                        t_l[11] = t_2[0]
                        t_l[12] = t_2[1]
                        t_l[13] = t_2[2]

                        #do action
                        rbs.arm_controller.move_to(position_list[i])

                        # t_n 为 next_state 变量
                        t_n = np.zeros((14),np.float32)
                        for j in range(len(position_list[i])):
                            t_n[j] = position_list[i+1][j]
                        
                        # 爪子实时获取
                        t_n[7] = rbs.gp[0]
                        # 爪子都设为0.1，最后一个动作为0.0
                        # if i == len(position_list)-2:
                        #     t_n[7] = 0.0
                        # else:
                        #     t_n[7] = 0.1
                        # 得到爪子position
                        t_1 = rbs.getCurrentGripperPose()
                        # 设置爪子position
                        t_n[8] = t_1[0]
                        t_n[9] = t_1[1]
                        t_n[10] = t_1[2]
                        # 得到cube的position
                        t_2 = rbs.getCurrentTargetPose()
                        # 设置cube的position
                        t_n[11] = t_2[0]
                        t_n[12] = t_2[1]
                        t_n[13] = t_2[2]

                        action = np.zeros((8),np.float32)
                        action[:-1] = t_n[:7] - t_l[:7]
                        if i == len(position_list)-2:
                            action[-1] = 0.0
                        else:
                            action[-1] = 0.1

                        # 获取reward
                        reward = rbs.getReward()
                        # 一步[state,action,reward,next_state,done]
                        data_list.append([t_l,action,reward,t_n,0.0])
                    # arm_group.go(wait = True)
                    # rbs.getDist()

                    # print xx,yy,zz
                    position = rbs.getCurrentGripperPose()
                    # print "Current Gripper Position is %s" % position
                    currentPos = np.array((position[0], position[1], position[2]))
                    target_position = rbs.getCurrentTargetPose()
                    # print "Current Target Cube Position is %s" % target_position
                    rbs.destPos = np.array((target_position[0], target_position[1], target_position[2]))
                    print u'-----------------------相差为---------------------',tx,ty,tz
                    display_trajectory_publisher = rospy.Publisher(
                                                    '/move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory)

                    print "============ Visualizing plan1"
                    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
                    display_trajectory.trajectory_start = robot.get_current_state()
                    display_trajectory.trajectory.append(trajectory)
                    display_trajectory_publisher.publish(display_trajectory)
                    group_variable_values = arm_group.get_current_joint_values()
                    print "============ Joint values: ", group_variable_values
                    #close gripper
                    close_gripper(gripper_goal)
                    grasping_client.tuck()
                    if_ac = rbs.getDist()
                    print '~~~~~~~~~~~~~~~~~~~~距离~~~~~~~~~~~~~~~',if_ac
                    if if_ac < 0.05:
                        print '!!!!!!!!!!!!!!!!!!!!success!!!!!!!!!!!!!!!!!!!!'
                        print "======================Current Target Cube Position is======================= %s" % target_position
                        print "======================================do action at=============================== ",xx,yy,zz
                        print u'-----------------------相差为---------------------',tx,ty,tz
                        success_l.append([tx,ty,tz])
                        for ii in data_list:
                            ii[-1] = 1.0
                        # time.sleep(5)
                        shit += 1
                    
                    
                    # time.sleep(1)
                    print 'start gripping action times :',fuck

                   
                    ###
                    si = ''
                    for i in data_list:
                        si = out_to_string(i)
                        f.write(si+'\n')
                    data_list = []
                print u'成功率:%d%%'%(100*(float(shit)/fuck))
                print u'成功列表为: '
                for i in success_l:
                    print i
                ia = 0.
                ib = 0.
                ic = 0.
                for i in success_l:
                    ia += i[0]
                    ib += i[1]
                    ic += i[2]
                print ia/len(success_l),ib/len(success_l),ic/len(success_l)
        elif mycmd == 'fo':
            xx += tx 
            yy += ty 
            zz += tz 
            print 'follow end ',xx,yy,zz
            pose_target = geometry_msgs.msg.Pose()
            pose_target.orientation.w = 1.0
            pose_target.position.x = xx #0.7
            pose_target.position.y = yy #-0.05
            pose_target.position.z = zz #1.1
            arm_group.set_pose_target(pose_target)
            trajectory = arm_group.plan()
            print type(trajectory)
            position_list = [i.velocities for i in trajectory.joint_trajectory.points]
            print len(trajectory.joint_trajectory.points)
            arm_group.go(wait = True)
            rbs.getDist()
            print xx,yy,zz
            display_trajectory_publisher = rospy.Publisher(
                                            '/move_group/display_planned_path',
                                            moveit_msgs.msg.DisplayTrajectory)

            print "============ Visualizing plan1"
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = robot.get_current_state()
            display_trajectory.trajectory.append(trajectory)
            display_trajectory_publisher.publish(display_trajectory)
            group_variable_values = arm_group.get_current_joint_values()
            print "============ Joint values: ", group_variable_values
        elif mycmd == 'f':
            print "============ Generating plan 1"
            xx = float(raw_input('x:'))
            yy = float(raw_input('y:'))
            zz = float(raw_input('z:'))

            pose_target = geometry_msgs.msg.Pose()
            pose_target.orientation.w = 1.0
            pose_target.position.x = xx #0.7
            pose_target.position.y = yy #-0.05
            pose_target.position.z = zz #1.1
            arm_group.set_pose_target(pose_target)
            trajectory = arm_group.plan()
            print type(trajectory)
            position_list = [i.positions for i in trajectory.joint_trajectory.points]
            # print trajectory.joint_names
            # trajectory.points.append(JointTrajectoryPoint())
            # print trajectory.joint_trajectory.points[0].positions
            # print trajectory.joint_trajectory.points[0].velocities
            # print trajectory.joint_trajectory.points[0].accelerations
            print '-------------------------------------------'
            for i in trajectory.joint_trajectory.points:
                print '-------------------------------------------'
                print "positions",i.positions
                print "velocities",i.velocities
                print "accelerations",i.accelerations
                print '-------------------------------------------'
            print '-------------------------------------------'
            for i in position_list[:-1]:
                # print 'move to ',i
                # rbs.change_arm_angle(i)
                rbs.arm_controller.move_to(i)
                rbs.getDist()
            
            # print type(trajectory.joint_trajectory.points)
            # print trajectory.joint_trajectory.points
            # arm_group.go(wait = True)
            display_trajectory_publisher = rospy.Publisher(
                                            '/move_group/display_planned_path',
                                            moveit_msgs.msg.DisplayTrajectory)

            print "============ Visualizing plan1"
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = robot.get_current_state()
            display_trajectory.trajectory.append(trajectory)
            display_trajectory_publisher.publish(display_trajectory)
            group_variable_values = arm_group.get_current_joint_values()
            print "============ Joint values: ", group_variable_values
            # group_variable_values[0] = 1.0
            # arm_group.set_joint_value_target(group_variable_values)

            # plan2 = arm_group.plan()
            # arm_group.go(wait = True)
        
        elif mycmd == 'mc':
            i = 0
            while  i < 100:
                i += 1
                rbs.change_arm_angle(velocities)
                rbs.getDist()
                time.sleep(1.5)
        elif mycmd == 'm':
            with open('arm_control.txt','w') as f:
                i = 0
                while  i < 20:
                    i += 1
                    f.write(str(velocities)+'\n')
                    grasping_client.my_arm_control(velocities)
                    rbs.getDist()
                    time.sleep(1)

        elif mycmd == 'g':
            control_gripper(gripper_goal)

        mycmd = raw_input("your commond: ")
        if mycmd == 'q':
            exit()

    while not rospy.is_shutdown():
        rospy.loginfo("Placing object...")
        pose = PoseStamped()
        pose.pose = cube.primitive_poses[0]
        pose.pose.position.z += 0.05
        pose.header.frame_id = cube.header.frame_id
        if grasping_client.place(cube, pose):
            break
        rospy.logwarn("Placing failed.")



    # Tuck the arm, lower the torso
    grasping_client.tuck()
    torso_action.move_to([0.0, ])
