import gym
import time
import numpy as np
import os
from os.path import expanduser

from gym import utils, spaces
from gym.utils import seeding

import sys
import random
from numpy import matrix
from math import cos, sin
from cv_bridge import CvBridge, CvBridgeError
import cv2
import psutil

from objectToPickUp import ObjectToPickUp
from kinect import Kinect



class SmartBotEnv(gym.Env):

    rewardSuccess = 500
    rewardFailure = 0
    rewardUnreachablePosition = -5
    # when set to True, the reward will be rewardSuccess if gripper could grasp the object, rewardFailure otherwise
    # when set to False, the reward will be calculated from the distance between the gripper & the position of success
    binaryReward = False
    
    # some algorithms (like the ddpg from /home/joel/Documents/gym-gazebo/examples/pincher_arm/smartbot_pincher_kinect_ddpg.py)
    # currently assume that the observation_space has shape (x,) instead of (220,300), so for those algorithms set this to True
    flattenImage = True
    
    # how many times reset() has to be called in order to move the ObjectToPickUp to a new (random) position
    randomPositionAtResetFrequency = 50
    resetCount = 0

    state = np.array([])
    imageCount = 0
    home = expanduser("~")
    imageWidth = 300
    imageHeight = 220
    boxLength = 0.07
    boxWidth = 0.03
    boxHeight = 0.03
    gripperDistance = 0.032 # between the fingers
    gripperHeight = 0.035 # from base to finger tip
    gripperWidth = 0.03
    gripperRadius = 0.2 # maximal distance between robot base and gripper on the floor

    def __init__(self):
        """ Initializes the environment. """

        # the state space (=observation space) are all possible depth images of the kinect camera
        if(self.flattenImage):
            self.observation_space = spaces.Box(low=0, high=255, shape=[self.imageHeight*self.imageWidth], dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=[self.imageHeight,self.imageWidth], dtype=np.uint8)

        # the action space are all possible positions & orientations (6-DOF),
        # which are bounded in the area in front of the robot arm where an object can lie (see reset())
        boundaries_xAxis = [0.04, 0.3]      # box position possiblities: (0.06, 0.22)
        boundaries_yAxis = [-0.25, 0.25]    # box position possiblities: (-0.2, 0.2)
        boundaries_phi = [0, np.pi]

        low = np.array([boundaries_xAxis[0], boundaries_yAxis[0], boundaries_phi[0]])
        high = np.array([boundaries_xAxis[1], boundaries_yAxis[1], boundaries_phi[1]])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.reward_range = (-np.inf, np.inf)
        
        self.seed()

        self.box = ObjectToPickUp(length = self.boxLength, width = self.boxWidth, height = self.boxHeight)
        self.kinect = Kinect(self.imageWidth, self.imageHeight, x=0.0, y=0.0, z=1.0)

    # __init__

    def seed(self, seed=None):
        """ Seeds the environment (for replicating the pseudo-random processes). """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    # seed

    def reset(self):
        """ Resets the state of the environment and returns an initial observation."""
        
        print("reset")
        self.box.place(randomPlacement=True)
        
        # get depth image
        image = self.kinect.getImage(self.box, filter=False, flattenImage=self.flattenImage, saveImage=True)

        self.state = image
        return self.state
    # reset

    def close(self):
        """ Closes the environment and shuts down the simulation. """
        print("closing SmartBotEnv")
        super(gym.Env, self).close()
    # close

    def step(self, action):
        """ Executes the action (i.e. moves the arm to the pose) and returns the reward and a new state (depth image). """

        # determine if gripper could grasp the ObjectToPickUp
        reward = self.calculateReward(action)

        self.box.place(randomPlacement=True)
        # print(self.box.pos)
        # print(self.box.phi)
        # print(self.box.a)
        # print(self.box.b)
        # print(self.box.c)
        # print(self.box.d)
        # get depth image
        image = self.kinect.getImage(self.box, filter=False, flattenImage=self.flattenImage, saveImage=True)

        self.state = image
        done = False
        info = {}

        return self.state, reward, done, info
    # step


    def getUnitVectorsFromOrientation(self, quaternion):
        """ Calculates the unit vectors (described in the /world coordinate system)
            of an object, which orientation is given by the quaternion. """
        explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(explicit_quat)

        rot_mat = tf.transformations.euler_matrix(roll, pitch, yaw)
        ex = np.dot(rot_mat, np.matrix([[1], [0], [0], [1]])) # e.g. [[0.800] [0.477] [-0.362] [1.]]
        ex = ex[:3] / ex[3]

        ey = np.dot(rot_mat, np.matrix([[0], [1], [0], [1]]))
        ey = ey[:3] / ey[3]

        ez = np.dot(rot_mat, np.matrix([[0], [0], [1], [1]]))
        ez = ez[:3] / ez[3]
        return ex, ey, ez
    # getUnitVectorsFromOrientation

    def calculateReward(self, action):
        """ Calculates the reward for the current timestep, according to the gripper position and the pickup position. 
            A high reward is given if the gripper could grasp the box (pickup) if it would close the gripper. """
        # TODO
        return 42


        pickup_position_old = np.matrix([[pickup_pose_old.position.x],[pickup_pose_old.position.y],[pickup_pose_old.position.z]])
        
        pickup_position = np.matrix([[pickup_pose.position.x],[pickup_pose.position.y],[pickup_pose.position.z]])
        # print("pickup_position:")
        # print(pickup_position) # e.g. [[0.1], [0.0], [0.04]]

        # check if the gripper has crashed into the ObjectToPickUp
        tolerance = 0.005
        if(pickup_position[0] < pickup_position_old[0] - tolerance or pickup_position[0] > pickup_position_old[0] + tolerance or
            pickup_position[1] < pickup_position_old[1] - tolerance or pickup_position[1] > pickup_position_old[1] + tolerance or 
            pickup_position[2] < pickup_position_old[2] - tolerance or pickup_position[2] > pickup_position_old[2] + tolerance):
            pass
            # print("********************************************* gripper crashed into the ObjectToPickUp! *********************************************")
            # return self.rewardFailure
        
        # check if gripper is in the correct position in order to grasp the object
        
        # dimensions of the ObjectToPickUp & the gripper (see the corresponding sdf/urdf files)
        pickup_xdim = 0.07
        pickup_ydim = 0.02
        pickup_zdim = 0.03
        gripper_width = 0.032 # between the fingers
        gripper_height = 0.035

        # calculating the unit vectors (described in the /world coordinate system) of the objectToPickUp::link
        ex, ey, ez = self.getUnitVectorsFromOrientation(pickup_pose.orientation)
        
        # corners of bounding box where gripper_right_position has to be
        p1 = pickup_position + pickup_xdim/2 * ex + pickup_ydim/2 * ey
        p2 = p1 - pickup_xdim * ex
        p4 = p1 + ey * (pickup_ydim/2 + gripper_width - pickup_ydim)
        p5 = p1 + ez * pickup_zdim
        # print("corners of bounding box where gripper_right_position has to be:")
        # print(p1) # e.g. [[0.135] [0.01 ] [0.04]]
        # print(p2) # e.g. [[0.065] [0.01 ] [0.04]]
        # print(p4) # e.g. [[0.135] [0.032] [0.04]]
        # print(p5) # e.g. [[0.135] [0.01 ] [0.07]]

        # the vectors of the bounding box where gripper_right_position has to be
        # transpose is necessary because np.dot can't handle two (3,1) matrices, so one of them has to be a (1,3) matrix
        u = np.transpose(p2 - p1) # e.g. [[ -7.00000000e-02  1.38777878e-17 -2.77555756e-17]]
        v = np.transpose(p4 - p1)
        w = np.transpose(p5 - p1)

        # corners of bounding box where gripper_left_position has to be
        p1_left = pickup_position + pickup_xdim/2 * ex - pickup_ydim/2 * ey
        p2_left = p1_left - pickup_xdim * ex
        p4_left = p1_left - ey * (pickup_ydim/2 + gripper_width - pickup_ydim)
        p5_left = p1_left + ez * pickup_zdim
        # print("corners of bounding box where gripper_left_position has to be:")
        # print(p1_left) # e.g. [[0.135] [-0.01 ] [0.04]]
        # print(p2_left) # e.g. [[0.065] [-0.01 ] [0.04]]
        # print(p4_left) # e.g. [[0.135] [-0.032] [0.04]]
        # print(p5_left) # e.g. [[0.135] [-0.01 ] [0.07]]

        # the vectors of the bounding box where gripper_left_position has to be
        u_left = np.transpose(p2_left - p1_left)
        v_left = np.transpose(p4_left - p1_left)
        w_left = np.transpose(p5_left - p1_left)

        graspSuccess = False
        # check if right gripper is on the right and left gripper is on the left of the ObjectToPickUp
        gripperRightIsRight = self.isPositionInCuboid(gripper_right_position, p1, p2, p4, p5, u, v, w)
        gripperLeftIsLeft = self.isPositionInCuboid(gripper_left_position, p1_left, p2_left, p4_left, p5_left, u_left, v_left, w_left)
        # check if right gripper is on the left and left gripper is on the right of the ObjectToPickUp
        gripperLeftIsRight = self.isPositionInCuboid(gripper_left_position, p1, p2, p4, p5, u, v, w)
        gripperRightIsLeft = self.isPositionInCuboid(gripper_right_position, p1_left, p2_left, p4_left, p5_left, u_left, v_left, w_left)
        # if one of the two scenarios is true, the grasping would be successful
        if((gripperRightIsRight and gripperLeftIsLeft) or (gripperLeftIsRight and gripperRightIsLeft)):
            print("********************************************* grasping would be successful! *********************************************")
            graspSuccess = True
        else:
            graspSuccess = False
        
        if(self.binaryReward):
            if(graspSuccess):
                return self.rewardSuccess
            else:
                return self.rewardFailure
        else:
            # calculate reward according to the distance from the gripper to the middle of the bounding box
            # pM = middle of the box where gripper_right_position has to be
            pM = p1 + 0.5 * np.transpose(u) + 0.5 * np.transpose(v) + 0.5 * np.transpose(w) # e.g. matrix([[0.1],[0.021],[0.055]])
            distance = np.linalg.norm(pM - gripper_right_position) # e.g. 0.1375784482561938
            # invert the distance, because smaller distance == closer to the goal == more reward
            reward = 1.0 / distance # e.g. 7.2685803094525
            # scale the reward if gripper is in the bounding box
            # (if gripper_right_position is exaclty at an edge of bounding box (e.g. p1), unscaled reward would be approx 25.2)
            if(graspSuccess):
                reward = 5 * reward
            # print("received reward: " + str(reward))
            return reward
        # if
    # calculateReward

    def isPositionInCuboid(self, gripper_position, p1, p2, p4, p5, u, v, w):
        """ Checks if gripper_position is in the correct location (i.e. within the cuboid described by u, v & w). """
        # (see https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d)
        if(np.dot(u, gripper_position) > np.dot(u, p1) and np.dot(u, gripper_position) < np.dot(u, p2)):
            # print("gripper is in correct position (x-axis)")
            if(np.dot(v, gripper_position) > np.dot(v, p1) and np.dot(v, gripper_position) < np.dot(v, p4)):
                # print("gripper is in correct position (y-axis)")
                if(np.dot(w, gripper_position) > np.dot(w, p1) and np.dot(w, gripper_position) < np.dot(w, p5)):
                    # print("gripper is in correct position (z-axis)")
                    return True
        return False
    # isPositionInCuboid

# class SmartBotEnv
