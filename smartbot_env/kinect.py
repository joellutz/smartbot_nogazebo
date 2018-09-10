import numpy as np
import os
from os.path import expanduser
import cv2

class Kinect():
    def __init__(self, width, height, x=0.0, y=0.0, z=0.0 ):
        self.width = width
        self.height = height
        self.image = np.array([])
        self.x = x
        self.y = y
        self.z = z
        self.imageCount = 0
        self.home = expanduser("~")
    # __init__

    def getImage(self, box, filter=False, flattenImage=False, saveImage=False, saveToFile=False):
        """ Reads the depth image from the kinect camera, adds Gaussian noise, crops, normalizes and saves it. """
        folder = self.home + "/Pictures/"
        # TODO: create depth image according to current box position
        return np.ones([self.width*self.height])


        # normalizing image
        cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        # adding noise to simulated depth image
        image = image + np.random.normal(0.0, 10.0, image.shape)
        # round to integer values and don't allow values <0 or >255 (necessary because of the added noise)
        image = np.clip(np.round(image), 0, 255).astype(np.uint8)
        if(setRandomPixelsToZero):
            # set approx. 5% of all pixels to zero
            mask = (np.random.uniform(0,1,size=image.shape) > 0.95).astype(np.bool)
            image[mask] = 0
        if(saveImage):
            # saving depth image
            if(saveToFile):
                np.set_printoptions(threshold="nan")
                f = open(folder + "depth_image_sim_" + str(self.imageCount) + ".txt", "w")
                print >>f, image
                f.close()
                # print("depth image saved as file")
            pathToImage = folder + "depth_image_sim_" + str(self.imageCount) + ".jpg"
            self.imageCount += 1
            try:
                cv2.imwrite(pathToImage, image)
                # print("depth image saved as jpg")
            except Exception as e:
                print(e)
        if(flattenImage):
            return image.flatten()
        else:
            return image
    # getImage

# class Kinect
