import cv2
import numpy as np

class Red_Detect(object):
    def __init__(self, image_w, image_h):
        self.width = image_w
        self.height = image_h
        
    def run(self, image, throttle):
        if image is None:
            return throttle, image
        
        red_threshold_low_1 = np.array([0, 30, 30])
        red_threshold_high_1 = np.array([10, 200, 200])
        
        red_threshold_low_2 = np.array([170, 30 ,30])
        red_threshold_high_2 = np.array([179, 200, 200])

        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        red_region_1 = cv2.inRange(image_hsv, red_threshold_low_1, red_threshold_high_1)
        red_region_2 = cv2.inRange(image_hsv, red_threshold_low_2, red_threshold_high_2)
        
        red_total = cv2.addWeighted(red_region_1, 1.0, red_region_2, 1.0, 0.0)
        num_red_pixels = cv2.countNonZero(red_total)
        
        if num_red_pixels > 250:
            return 0, image
        else:
            return throttle, image
        
            
            
