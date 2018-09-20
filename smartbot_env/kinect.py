import numpy as np
import os
from os.path import expanduser
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import logging
from scipy.signal import medfilt2d

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

    def getImage(self, box, filter=False, flattenImage=False, saveImage=False, saveToFile=False, folder="/Pictures/"):
        """ Reads the depth image from the (pseudo) kinect camera. """
        # TODO: change image based on camera location (x,y,z), add filter, better image sizing capabilities (height & width)

        plt.ioff()
        fig, ax = plt.subplots(frameon=False, figsize=(8,4), dpi=self.width/8)
        # pixels = size_inches * dpi
        # logging.info(fig.get_size_inches())
        # logging.info(fig.dpi)
        
        ax.axes.set_xlim([-box.gripperRadius - box.length, box.gripperRadius + box.length])
        ax.axes.set_ylim([0 - box.length, box.gripperRadius + box.length])
        ax.axes.set_aspect('equal', adjustable='box')
        fig.tight_layout(pad=0)
        ax.set_axis_off()

        rect = patches.Polygon([box.a, box.b, box.c, box.d], closed=True, color="black")
        ax.add_patch(rect)
        
        # plt.ion()
        # plt.show()

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # logging.info(image.shape) # e.g. (153600,)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # logging.info(image.shape) # e.g. (160, 320, 3)
        image = self.rgb2gray(image)
        # logging.info(image.shape) # (160, 320)

        # # Save as png (not possible before converting it to a numpy array, as then reshaping wouldn't work!)
        # plt.savefig("testfig")

        plt.close(fig)

        # adding noise to simulated depth image
        image = image + np.random.normal(0.0, 10.0, image.shape)
        image = medfilt2d(image, kernel_size=5)

        # # round to integer values and don't allow values <0 or >255 (necessary because of the added noise)
        # image = np.clip(np.round(image), 0, 255).astype(np.uint8)
        self.imageCount += 1

        if(saveImage and self.imageCount % 100 == 0):
            folder = self.home + folder
            # saving depth image
            if(saveToFile):
                np.set_printoptions(threshold="nan")
                f = open(folder + "depth_image_sim_" + str(self.imageCount) + ".txt", "w")
                print >>f, image
                f.close()
                # logging.info("depth image saved as file")
            pathToImage = folder + "depth_image_sim_" + str(self.imageCount) + ".jpg"
            try:
                cv2.imwrite(pathToImage, image)
                # logging.info("depth image saved as jpg")
            except Exception as e:
                logging.info(e)
        if(flattenImage):
            return image.flatten()
        else:
            return image
    # getImage

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    # rgb2gray

# class Kinect
