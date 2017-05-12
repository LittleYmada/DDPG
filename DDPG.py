import ACNet
from ACNet import Poliy
from ACNet import QValue
from ACNet import ReplayBuffer
from ACNet import OUProcess
import numpy as np
import random
from collections import deque

import keras.backend as K
import tensorflow as tf
from keras.engine.training import *
from keras.layers import Dense, Input, merge
from keras.optimizers import Adam
import demo


import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json
import timeit


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
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from moveit_msgs.msg import PlaceLocation, MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

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

QLEARNINGRATE=0.05
PLEARNINGRATE=0.05
BATCHSIZE=100
#[[angles],open_or_close]
ACTIONSIZE=8
#[[angles],[x_clip,y_clip,z_clip],open_or_close,[x_item,y_item,z_item]]
STATESIZE=14

OU = OU()       #Ornstein-Uhlenbeck Process
#tray
waypoints = []
lim=[[1.60,1.60],[1.51,1.22],[3.14,3.14],[2.25,2.25],[3.14,3.14],[2.18,2.18],[3.14,3.14]]
#----------------------------groups----------------------------
arm_group = moveit_commander.MoveGroupCommander("arm")
gripper_group = moveit_commander.MoveGroupCommander("gripper")
all_group = moveit_commander.MoveGroupCommander("arm_with_torso")
#----------------------------groups-----------------------------


def playGame(train_indicator=0):    #1 means Train, 0 means simply Run

#######################################################
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

    rbs =demo.robotGame()
    
    image_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image)
    # depth_img = message_filters.Subscriber('/head_camera/depth_downsample/image_raw', Image)
    # info_sub = message_filters.Subscriber('camera_info', CameraInfo)
    # Setup clients
    move_base = demo.MoveBaseClient()
    torso_action = demo.FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    head_action = demo.PointHeadClient()
    grasping_client = demo.GraspingClient()

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
    
##############################################################

    cmd=raw_input("which mod do you want to learn on (off/on line)")
    if cmd=="on":
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 7  #Steering/Acceleration/Brake
    state_dim = 13  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 50
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    '''config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)'''
    sess = tf.Session()
    from keras import backend as K
    K.set_session(sess)
    policy = Poliy(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    value = QValue(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environmentenv = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    #Now load the weight
    print "Now we load the weight"
    try:
        policy.model.load_weights("actormodel.h5")
        value.model.load_weights("criticmodel.h5")
        policy.target_model.load_weights("actormodel.h5")
        value.target_model.load_weights("criticmodel.h5")
        print "Weight load successfully"
    except:
        print "Cannot find the weight"

    print "Ros Fetch Process Start."
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        rbs.reset()
        target_position=rbs.getCurrentTargetPose()
        time.sleep(1)
        while target_position[2] < 0.5:
             print target_position[2]
             rbs.reset()
             time.sleep(2)
         print "Reset Success"
         if rbs.getDist()<0.01: continue
         demo.open_gripper(gripper_goal)
         s_t_ori=rbs.getState()

#get state
        s_t = np.array(s_t_ori)
     
        total_reward = 0.
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = policy.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.getNoise(a_t_original[0][0],  0.0 , 1.00, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.getNoise(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.getNoise(a_t_original[0][2], -0.1 , 1.00, 0.05)
            noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.getNoise(a_t_original[0][3],  0.0 , 1.00, 0.30)
            noise_t[0][4] = train_indicator * max(epsilon, 0) * OU.getNoise(a_t_original[0][4],  0.5 , 1.00, 0.10)
            noise_t[0][5] = train_indicator * max(epsilon, 0) * OU.getNoise(a_t_original[0][5], -0.1 , 1.00, 0.05)
            noise_t[0][6] = train_indicator * max(epsilon, 0) * OU.getNoise(a_t_original[0][6],  0.0 , 1.00, 0.30)
            

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            a_t[0][3] = a_t_original[0][3] + noise_t[0][3]
            a_t[0][4] = a_t_original[0][4] + noise_t[0][4]
            a_t[0][5] = a_t_original[0][5] + noise_t[0][5]
            a_t[0][6] = a_t_original[0][6] + noise_t[0][6]
            
            a_t_t=[
                            (lim[0][0]  if a_t[0][0] > 0 else lim[0][1])*a_t[0][0],
                            (lim[1][0]  if a_t[0][1] > 0 else lim[1][1])*a_t[0][1],
                            (lim[2][0]  if a_t[0][2] > 0 else lim[2][1])*a_t[0][2],
                            (lim[3][0]  if a_t[0][3] > 0 else lim[3][1])*a_t[0][3],
                            (lim[4][0]  if a_t[0][4] > 0 else lim[4][1])*a_t[0][4],
                            (lim[5][0]  if a_t[0][5] > 0 else lim[5][1])*a_t[0][5],
                            (lim[6][0]  if a_t[0][6] > 0 else lim[6][1])*a_t[0][6]
            ]
             jvv=[]
             for i in xrange(7):
             	jvv.append(a_t_t[i]+s_t[i])
             rbs.setJointValues(jvv)
             rbs.arm_controller.move_to(rbs.joint_positions)
             time.sleep(1)
             s_t1=rbs.getState()
             r_t=rbs.getReward()
             if rbs.getDist()<0.01:
             	done=1
             else:
             	done=0
            #ob, r_t, done, info = env.step(a_t[0])
        
            buff.add_sequence(s_t, a_t_t, r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getbatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = value.target_model.predict([new_states, policy.target_model.predict(new_states)])  
           
            for k in xrange(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += value.model.train_on_batch([states,actions], y_t) 
                a_for_grad = policy.model.predict(states)
                grads = value.gradients(states, a_for_grad)
                policy.train(states, grads)
                policy.target_train()
                value.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    #env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
