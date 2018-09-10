import gym
import time
import numpy as np
import matplotlib.path as mplPath


from gym import utils, spaces
from gym.utils import seeding

import sys
import random

from objectToPickUp import ObjectToPickUp
from kinect import Kinect



class SmartBotEnv(gym.Env):

    rewardSuccess = 500
    rewardFailure = 0
    rewardUnreachablePosition = -5 # TODO: do we need this anymore?
    # when set to True, the reward will be rewardSuccess if gripper could grasp the object, rewardFailure otherwise
    # when set to False, the reward will be calculated from the distance between the gripper & the position of success
    binaryReward = True
    
    # some algorithms (like the ddpg from /home/joel/Documents/gym-gazebo/examples/pincher_arm/smartbot_pincher_kinect_ddpg.py)
    # currently assume that the observation_space has shape (x,) instead of (220,300), so for those algorithms set this to True
    flattenImage = True

    state = np.array([])
    imageWidth = 300 # TODO: tbd
    imageHeight = 220 # TODO: tbd
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

        # TODO: maybe don't allow positions outside the radius of the gripper (i.e. unreachable positions)
        boundaries_xAxis = [-self.gripperRadius, self.gripperRadius]
        boundaries_yAxis = [0, self.gripperRadius]
        boundaries_phi = [0, np.pi]

        low = np.array([boundaries_xAxis[0], boundaries_yAxis[0], boundaries_phi[0]])
        high = np.array([boundaries_xAxis[1], boundaries_yAxis[1], boundaries_phi[1]])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.reward_range = (-np.inf, np.inf)
        
        self.seed()

        # create object for box (object to pick up)
        self.box = ObjectToPickUp(length = self.boxLength, width = self.boxWidth, height = self.boxHeight, gripperRadius=self.gripperRadius)
        # create object for kinect
        self.kinect = Kinect(self.imageWidth, self.imageHeight, x=0.0, y=0.0, z=1.0)

    # __init__

    def seed(self, seed=None):
        """ Seeds the environment (for replicating the pseudo-random processes). """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    # seed

    def reset(self):
        """ Resets the state of the environment and returns an initial observation."""
        # place box
        self.box.place(randomPlacement=True)

        # for testing purposes
        # self.box.place(randomPlacement=False, x=0.0, y=0.0, phi=0.0)
        
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
        gripperX = action[0].astype(np.float64)
        gripperY = action[1].astype(np.float64)
        gripperPhi = action[2].astype(np.float64)

        # for testing purposes
        # gripperX = self.boxLength/2
        # gripperY = self.gripperDistance/2 - 0.01
        # gripperPhi = np.pi/2

        reward = self.calculateReward(gripperX, gripperY, gripperPhi)
        print("received reward: " + str(reward))

        # re-place object to pick up
        self.box.place(randomPlacement=True)
        # get depth image
        image = self.kinect.getImage(self.box, filter=False, flattenImage=self.flattenImage, saveImage=True)

        self.state = image
        done = False
        info = {}

        return self.state, reward, done, info
    # step

    def calculateReward(self, gripperX, gripperY, gripperPhi):
        """ Calculates the reward for the current timestep, according to the gripper position and the pickup position. 
            A high reward is given if the gripper could grasp the box (pickup) if it would close the gripper. """
        # TODO: Calculate reward based on current box position and chosen action
        # return 42

        # Calculate corner points of non-rotated gripper
        leftX = gripperX - self.gripperDistance/2
        rightX = gripperX + self.gripperDistance/2
        topY = gripperY + self.gripperWidth/2
        bottomY = gripperY - self.gripperWidth/2
        ag = np.array([leftX, topY])
        bg = np.array([leftX, bottomY])
        cg = np.array([rightX, bottomY])
        dg = np.array([rightX, topY])

        # Rotate corner points around point (position or gripper)
        gripperPos = np.array([gripperX, gripperY])
        cos, sin = np.cos(gripperPhi), np.sin(gripperPhi)
        R = np.array([[cos, -sin], [sin, cos]])
        ag = gripperPos + R.dot(ag - gripperPos)
        bg = gripperPos + R.dot(bg - gripperPos)
        cg = gripperPos + R.dot(cg - gripperPos)
        dg = gripperPos + R.dot(dg - gripperPos)

        # one finger is between ag & bg, the other finger is between cg & dg

        # print("Current gripper position:")
        # print("Position: {0}\nPhi: {1}\na: {2}\nb: {3}\nc: {4}\nd: {5}\n".format((gripperPos),
        #     (gripperPhi), (ag), (bg), (cg), (dg)))

        # print("Current box position:")
        # print(self.box)


        # check if center of gravity of box is between the gripper fingers (i.e. inside the ag-bg-cg-dg polygon)
        # see e.g.: https://stackoverflow.com/a/23453678
        bbPath_gripper = mplPath.Path(np.array([ag, bg, cg, dg]))
        cogBetweenFingers = bbPath_gripper.contains_point((self.box.pos[0], self.box.pos[1]))
        print("center of gravity is between the fingers: {}".format(cogBetweenFingers))

        # check if both gripper fingers don't intersect with the box
        bbPath_box = mplPath.Path(np.array([self.box.a, self.box.b, self.box.c, self.box.d]))
        bbPath_gripper_left = mplPath.Path(np.array([ag, bg]))
        bbPath_gripper_right = mplPath.Path(np.array([cg, dg]))
        leftGripperCrashes = bbPath_box.intersects_path(bbPath_gripper_left, filled=True)
        rightGripperCrashes = bbPath_box.intersects_path(bbPath_gripper_right, filled=True)
        print("left gripper crashes: {}".format(leftGripperCrashes))
        print("right gripper crashes: {}".format(rightGripperCrashes))

        # if the center of gravity of the box is between the gripper fingers and none of the fingers collide with the box, we are able to grasp the box
        if(cogBetweenFingers and not leftGripperCrashes and not rightGripperCrashes):
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
            # calculate reward according to the distance from the gripper to the center of gravity of the box
            distance = np.linalg.norm(gripperPos - self.box.pos) # e.g. 0.025
            # invert the distance, because smaller distance == closer to the goal == more reward
            reward = 1.0 / distance # e.g. 40
            # scale the reward if grasping would be successful
            if(graspSuccess):
                reward = 5 * reward
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
