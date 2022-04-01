import cv2
import numpy as np
import jetson.inference
import jetson.utils

from matplotlib import cm
import os
import sys

class Preprocess_image(object):
    def __init__(self,image_w, image_h):
        self.width = image_w
        self.height = image_h
    
    def run(self, img_arr):
        # image must be 3-dimension, only upper part vanish proprocessing is availiable
        if img_arr is None:
            print("no image!!")
            return img_arr

        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        mask = np.zeros_like(gray)
        polygon = np.array([[(0, self.height*1/2), (self.width, self.height*1/2), (self.width, self.height), (0, self.height)]], np.int32)
        cv2.fillPoly(mask, polygon, 255)

        results = np.zeros_like(img_arr)
        
        results[:, :, 0] = cv2.bitwise_and(img_arr[:, :, 0], mask)
        results[:, :, 1] = cv2.bitwise_and(img_arr[:, :, 1], mask)
        results[:, :, 2] = cv2.bitwise_and(img_arr[:, :, 2], mask)
        
        return results

    
