import numpy as np
import cv2
import jetson.inference
import jetson.utils

from matplotlib import cm
import os

class CarDetector(object):

    def __init__(self, show_bounding_box, threshold, debug=False):
        # self.car_cascade = cv2.CascadeClassifier("/home/autocar2/vehicle_detection_haarcascades/car.xml")
        self.show_bounding_box = show_bounding_box
        self.debug = debug

        self.output_URI = ""
        self.network = "ssd-mobilenet-v2"
        self.overlay = "box, labels, conf" if self.show_bounding_box else ""
        self.threshold = threshold

        self.is_headless = [""]  # ["--headless"] # if sys.argv[0].find('console.py') != -1 else [""]
        self.my_model = ["--model=/home/autocar5/projects/donkeycar/donkeycar/parts/model/roboflow_s/ssd-mobilenet.onnx"]
        self.my_label = ["--labels=/home/autocar5/projects/donkeycar/donkeycar/parts/model/roboflow_s/labels.txt"]
        self.my_input_blob = ["--input-blob=input_0"]
        self.my_output_cvg = ["--output-cvg=scores"]
        self.my_output_bbox = ["--output-bbox=boxes"]

        self.output = jetson.utils.videoOutput(self.output_URI, self.is_headless)
        self.net = jetson.inference.detectNet(self.network, self.my_model + self.my_label + self.my_input_blob + self.my_output_cvg + self.my_output_bbox, self.threshold)
        self.detected_objects = 0

    """
    def convertImageArrayToPILImage(self, img_arr):
        img = Image.fromarray(img_arr.astype('uint8'), 'RGB')
        return img

    '''
    Return an object if there is a traffic light in the frame
    '''
    def detect_car (self, cuda_img):
        # img = self.convertImageArrayToPILImage(img_arr)
        # gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # cars = self.car_cascade.detectMultiScale(gray, 1.1, 6) # scale factor, minNeighbors

        # car_detected = None
        is_detected = False

        detections = self.net.Detect(cuda_img, overlay=self.overlay)

        # print the detections
        print("detected {:d} objects in image".format(len(detections)))
        if len(detections) > 0: is_detected = True

        return

        if cars != None:
            self.cars = cars
            is_detected = True

        # if car_detected:
        #     self.last_5_scores.append(car_detected.score)
        #     sum_of_last_5_score = sum(list(self.last_5_scores))
        #     # print("sum of last 5 score = ", sum_of_last_5_score)

        #     if sum_of_last_5_score > self.LAST_5_SCORE_THRESHOLD:
        #         return car_detected
        #     else:
        #         print("Not reaching last 5 score threshold")
        #         return None
        # else:
        #     self.last_5_scores.append(0)
        #     return None

        return is_detected

    def draw_bounding_box(self, cars, img_arr):
        for (x, y, w, h) in cars:
            cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 0, 255), 2)
    """

    def run(self, img_arr, throttle, debug=False):

        # img_arr to cuda_img
        if img_arr is None:
            return throttle, img_arr
        else: cuda_img = jetson.utils.cudaFromNumpy(img_arr)

        # detect objects in the image (with overlay)
        detections = self.net.Detect(cuda_img, overlay=self.overlay)
        for detection in detections:
            print(detection.ClassID)
        '''
        for dectection in detections:
            if detection == 'car':
                self.detected_objects +1
            print(detection) # <class 'jetson.inference.detectNet.Detection'>
        '''

        # print the detections
        # print("detected {:d} objects in image".format(self.detected_objects))
        '''
        for detection in detections:
            (x, y, w, h) = (detection.Left, detection.Top, detection.Width, detection.Height)
            cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('img',img_arr)
        cv2.waitKey(0)
        '''
        
        # img_arr = jetson.utils.cudaToNumpy(cuda_img)
        if self.detected_objects > 0:
            return 0, img_arr
        else:
            return throttle, img_arr
