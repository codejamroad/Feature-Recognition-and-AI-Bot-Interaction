import cv2
import time
import numpy as np
import random

def calculate_points(input_frames):

    protoFile = "hand/pose_deploy.prototxt"
    weightsFile = "hand/pose_iter_102000.caffemodel"
    nPoints = 21
    threshold = 0.2
    width = 640
    height = 480

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    k=0        
    #Start landmark extraction only on selected frames
    for frame in input_frames:
        t = time.time()
        k = k+1
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (width,height), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            if i==1: #to skip the extra point in the trained model
                continue
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (width, height))
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > threshold :
                points.append(int(point[0]))
                points.append(int(point[1]))
                #cv2.circle(frame, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            else :
                points.append(0) #X
                points.append(0) #Y

            #Uncomment the below line to see output
            #cv2.imshow('Output-Skeleton', frame)
            #key = cv2.waitKey(1)
            #if key == 27:
                #break
        print("Time Taken for frame {}  = {}".format(k, time.time() - t))
    return points