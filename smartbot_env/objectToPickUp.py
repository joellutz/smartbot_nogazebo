import numpy as np
import random

class ObjectToPickUp():
    def __init__(self, length, width, height, x=0.0, y=0.0, phi=0.0):
        self.length = length
        self.width = width
        self.height = height
        self.pos = np.array([x, y])
        self.phi = phi
        self.a, self.b, self.c, self.d = self.calculateCornerPoints()
    # __init__

    def place(self, randomPlacement=True, x=0.0, y=0.0, phi=0.0):
        if(randomPlacement):
            print("setting random position of object to pick up")
            x = random.uniform(0.06, 0.22)
            y = random.uniform(-0.2, 0.2)
            self.phi = random.uniform(0, np.pi)
        else:
            print("setting non-random position of object to pick up")
            self.phi = phi
        self.pos = np.array([x, y])
        self.a, self.b, self.c, self.d = self.calculateCornerPoints()
        return self
    # place

    def calculateCornerPoints(self):
        # Calculate corner points of non-rotated object
        leftX = self.pos[0] - self.length/2
        rightX = self.pos[0] + self.length/2
        topY = self.pos[1] + self.width/2
        bottomY = self.pos[1] - self.width/2
        a = np.array([leftX, topY])
        b = np.array([leftX, bottomY])
        c = np.array([rightX, bottomY])
        d = np.array([rightX, topY])

        # Rotate corner points around point (position)
        cos, sin = np.cos(self.phi), np.sin(self.phi)
        R = np.array([[cos, -sin], [sin, cos]])
        v = a - self.pos
        matr = R.dot(v)
        a = self.pos + R.dot(a - self.pos)
        b = self.pos + R.dot(b - self.pos)
        c = self.pos + R.dot(c - self.pos)
        d = self.pos + R.dot(d - self.pos)
        return a, b, c, d

# class ObjectToPickUp