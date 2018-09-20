#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time

def demo(a):
    plt.cla()
    y = [xt*a+1 for xt in x]
    ax.set_ylim([0,15])
    ax.plot(x,y)

if __name__ == '__main__':
    plt.ion()
    fig, ax = plt.subplots()
    x = range(5)
    for a in range(1,4):
        demo(a)
        plt.pause(3)
        plt.draw()