#!/usr/bin/env python

import gym
import smartbot_env
import random
from smartbot_env.objectToPickUp import ObjectToPickUp
from smartbot_env.kinect import Kinect
import numpy as np


if __name__ == '__main__':
    
    # env = gym.make("SmartBotEnv-v0")
    # env.seed()

    imageWidth = 320 # TODO: tbd
    imageHeight = 160 # TODO: tbd
    boxLength = 0.07
    boxWidth = 0.03
    boxHeight = 0.03
    gripperDistance = 0.032 # between the fingers
    gripperHeight = 0.035 # from base to finger tip
    gripperWidth = 0.03
    gripperRadius = 0.2 # maximal distance between robot base and gripper on the floor

    # create object for box (object to pick up)
    box = ObjectToPickUp(length = boxLength, width = boxWidth, height = boxHeight, gripperRadius = gripperRadius)
    box.place(randomPlacement=False, x=0.1, y=0.1, phi=np.pi/8)

    # create object for kinect
    kinect = Kinect(imageWidth, imageHeight, x=0.0, y=0.0, z=1.0)
    image = kinect.getImage(box, saveImage=True)

    for i in range(100):
        print("getting image {}".format(i))
        box.place(randomPlacement=True)
        kinect.getImage(box, saveImage=True)


# if __main___