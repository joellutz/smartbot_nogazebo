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
from gripper import Gripper
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib



class SmartBotEnv(gym.Env):

    rewardSuccess = 500
    rewardFailure = 0
    rewardUnreachablePosition = -5 # TODO: do we need this anymore?
    # when set to True, the reward will be rewardSuccess if gripper could grasp the object, rewardFailure otherwise
    # when set to False, the reward will be calculated from the distance between the gripper & the position of success
    binaryReward = False
    
    # some algorithms (like the ddpg from /home/joel/Documents/gym-gazebo/examples/pincher_arm/smartbot_pincher_kinect_ddpg.py)
    # currently assume that the observation_space has shape (x,) instead of (220,300), so for those algorithms set this to True
    flattenImage = False

    state = np.array([])
    imageWidth = 320
    imageHeight = 160
    boxLength = 0.02
    boxWidth = 0.02
    boxHeight = 0.03
    gripperDistance = 0.032 # between the fingers
    gripperHeight = 0.035 # from base to finger tip
    gripperWidth = 0.03
    gripperRadiusMax = 0.2 # maximal distance between robot base and gripper on the floor
    gripperRadiusMin = 0.04 # minimal distance between robot base and gripper on the floor

    firstRender = True

    def __init__(self):
        """ Initializes the environment. """

        # the state space (=observation space) are all possible depth images of the kinect camera
        if(self.flattenImage):
            self.observation_space = spaces.Box(low=0, high=255, shape=[self.imageHeight*self.imageWidth], dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=[self.imageHeight,self.imageWidth], dtype=np.uint8)

        boundaries_r = [self.gripperRadiusMin, self.gripperRadiusMax]
        boundaries_phi = [0, np.pi]

        low = np.array([boundaries_r[0], boundaries_phi[0]])
        high = np.array([boundaries_r[1], boundaries_phi[1]])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.reward_range = (-np.inf, np.inf)
        
        self.seed()

        # create object for box (object to pick up)
        self.box = ObjectToPickUp(length = self.boxLength, width = self.boxWidth, height = self.boxHeight, gripperRadius=self.gripperRadiusMax)
        # create object for kinect
        self.kinect = Kinect(self.imageWidth, self.imageHeight, x=0.0, y=0.0, z=1.0)
        # create object for gripper
        self.gripper = Gripper(self.gripperDistance, self.gripperWidth, self.gripperHeight, r=0.0, phi=0.0)

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
        self.box.place(randomPlacement=False, x=0.0, y=0.1, phi=np.pi/2)
        
        # get depth image
        image = self.kinect.getImage(self.box, filter=False, flattenImage=self.flattenImage, saveImage=True)

        self.state = image
        return self.state
    # reset

    def close(self):
        """ Closes the environment and shuts down the simulation. """
        logging.info("closing SmartBotEnv")
        super(gym.Env, self).close()
    # close

    stepcount = 0
    winkel = np.linspace(0, np.pi)

    def step(self, action):
        """ Executes the action (i.e. moves the arm to the pose) and returns the reward and a new state (depth image). """

        # determine if gripper could grasp the ObjectToPickUp
        gripperR = action[0].astype(np.float64)
        gripperPhi = action[1].astype(np.float64)

        # # for testing purposes
        # gripperR = 0.1
        # gripperPhi = self.winkel[self.stepcount]
        # self.stepcount += 1

        self.gripper.place(gripperR, gripperPhi)

        logging.debug("moving arm to position: [{0} {1}]".format(gripperR, gripperPhi))
        # logging.debug("box position: {0}, {1}, {2}".format(self.box.pos[0], self.box.pos[1], self.box.phi))

        reward, graspSuccess = self.calculateReward()
        logging.debug("received reward: " + str(reward))

        # re-place object to pick up if grasp was successful
        # if(graspSuccess):
        #     self.box.place(randomPlacement=True)
        
        # get depth image
        image = self.kinect.getImage(self.box, filter=False, flattenImage=self.flattenImage, saveImage=True)

        self.state = image
        done = graspSuccess
        info = {}

        return self.state, reward, done, info
    # step

    def calculateReward(self):
        """ Calculates the reward for the current timestep, according to the gripper position and the pickup position. 
            A high reward is given if the gripper could grasp the box (pickup) if it would close the gripper. """

        # check if center of gravity of box is between the gripper fingers (i.e. inside the ag-bg-cg-dg polygon)
        # see e.g.: https://stackoverflow.com/a/23453678
        bbPath_gripper = mplPath.Path(np.array([self.gripper.a, self.gripper.b, self.gripper.c, self.gripper.d]))
        # one finger is between a & b, the other finger is between c & d
        cogBetweenFingers = bbPath_gripper.contains_point((self.box.pos[0], self.box.pos[1]))
        logging.debug("center of gravity is between the fingers: {}".format(cogBetweenFingers))

        # check if both gripper fingers don't intersect with the box
        bbPath_box = mplPath.Path(np.array([self.box.a, self.box.b, self.box.c, self.box.d]))
        bbPath_gripper_left = mplPath.Path(np.array([self.gripper.a, self.gripper.b]))
        bbPath_gripper_right = mplPath.Path(np.array([self.gripper.c, self.gripper.d]))
        leftGripperCrashes = bbPath_box.intersects_path(bbPath_gripper_left, filled=True)
        rightGripperCrashes = bbPath_box.intersects_path(bbPath_gripper_right, filled=True)
        logging.debug("left gripper crashes: {}".format(leftGripperCrashes))
        logging.debug("right gripper crashes: {}".format(rightGripperCrashes))

        # if the center of gravity of the box is between the gripper fingers and none of the fingers collide with the box, we are able to grasp the box
        if(cogBetweenFingers and not leftGripperCrashes and not rightGripperCrashes):
            logging.info("********************************************* grasping would be successful! *********************************************")
            graspSuccess = True
        else:
            graspSuccess = False
        
        if(self.binaryReward):
            if(graspSuccess):
                return self.rewardSuccess, graspSuccess
            else:
                return self.rewardFailure, graspSuccess
        else:
            # calculate reward according to the distance from the gripper to the center of gravity of the box
            distance = np.linalg.norm(self.gripper.pos - self.box.pos) # e.g. 0.025
            # invert the distance, because smaller distance == closer to the goal == more reward
            reward = 1.0 / (2 * distance)
            # scale the reward if grasping would be successful
            if(graspSuccess):
                reward = 50 * reward
            # elif(leftGripperCrashes or rightGripperCrashes): # "punishement" for crashing into the box
            #     reward = reward / 5
            reward = min(reward, 1000)
            return reward, graspSuccess
        # if
    # calculateReward

    def render(self, mode='human'):
        if(self.firstRender):
            # display stuff
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.axis("equal")
            self.ax.set_xlim([-self.box.gripperRadius - self.box.length - 0.2, self.box.gripperRadius + self.box.length + 0.2])
            self.ax.set_ylim([0 - self.box.length - 0.2, self.box.gripperRadius + self.box.length + 0.2])
            self.gripperLeftPoly = patches.Polygon([self.gripper.a, self.gripper.b], closed=True, color="black")
            self.gripperRightPoly = patches.Polygon([self.gripper.c, self.gripper.d], closed=True, color="black")
            self.pickMeUpPoly = patches.Polygon([self.box.a, self.box.b, self.box.c, self.box.d], closed=True, color="red")
            self.ax.add_artist(self.gripperLeftPoly)
            self.ax.add_artist(self.gripperRightPoly)
            self.ax.add_artist(self.pickMeUpPoly)
            self.firstRender = False
        # if
        plt.ion()
        plt.cla()
        self.gripperLeftPoly.set_xy([self.gripper.a, self.gripper.b])
        self.gripperRightPoly.set_xy([self.gripper.c, self.gripper.d])
        self.pickMeUpPoly.set_xy([self.box.a, self.box.b, self.box.c, self.box.d])
        self.ax.add_artist(self.gripperLeftPoly)
        self.ax.add_artist(self.gripperRightPoly)
        self.ax.add_artist(self.pickMeUpPoly)
        # self.fig.canvas.draw()
        plt.pause(0.0001)
        plt.draw()
    # render

# class SmartBotEnv
