import cv2 as cv
import numpy as np


def get_limits(color):
    c = np.uint8([[color]])
    hsv = cv.cvtColor(c, cv.COLOR_BGR2HSV)

    lowerLim = hsv[0][0][0] - 10, 100, 100
    upperLim = hsv[0][0][0] - 10, 225, 225

    lowerLim = np.array(lowerLim, dtype=np.uint8)
    upperLim = np.array(upperLim, dtype=np.uint8)

    return lowerLim, upperLim
