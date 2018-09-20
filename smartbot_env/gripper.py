import numpy as np
import random
import logging

class Gripper():
    def __init__(self, distance, width, height, r=0.0, phi=0.0):
        self.distance = distance
        self.width = width
        self.height = height
        self.place(r,phi)
    # __init__

    def place(self, r=0.0, phi=0.0):
        self.r = r
        self.phi = phi
        self.pos = self.calculateCartesianPos()
        self.a, self.b, self.c, self.d = self.calculateCornerPoints()
        return self
    # place

    def calculateCornerPoints(self):
        # Calculate corner points of non-rotated gripper TODO: very similar to ObjectToPickUp class, maybe make a super class for this?
        leftX = self.pos[0] - self.distance/2
        rightX = self.pos[0] + self.distance/2
        topY = self.pos[1] + self.width/2
        bottomY = self.pos[1] - self.width/2
        a = np.array([leftX, topY])
        b = np.array([leftX, bottomY])
        c = np.array([rightX, bottomY])
        d = np.array([rightX, topY])
        # one finger is between a & b, the other finger is between c & dyy

        # Rotate corner points around point (position)
        alpha = self.phi - np.pi/2 # necessary because phi=0 means far right of the plane
        cos, sin = np.cos(alpha), np.sin(alpha)
        R = np.array([[cos, -sin], [sin, cos]])
        a = self.pos + R.dot(a - self.pos)
        b = self.pos + R.dot(b - self.pos)
        c = self.pos + R.dot(c - self.pos)
        d = self.pos + R.dot(d - self.pos)
        return a, b, c, d
    # calculateCornerPoints

    def calculateCartesianPos(self):
        x = self.r * np.cos(self.phi)
        y = self.r * np.sin(self.phi)
        return np.array([x,y])
    # calculateCartesianPos

    def __str__(self):
        return "Position: {0}\nPhi: {1}\na: {2}\nb: {3}\nc: {4}\nd: {5}\n".format((self.pos),
            (self.phi), (self.a), (self.b), (self.c), (self.d))
    # __str__

# class ObjectToPickUp
