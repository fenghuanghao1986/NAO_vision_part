# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 20:18:29 2018

@author: fengh
"""

# import packages
# deque is list-like container with fast appends and pops on either end
from collections import deque
import numpy as np
# argparse module makes it easy to write user-fridenly command-line interface
# it automatically generates help/usage messages/errors inputs invalid
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
# handle parsing two command line arguments
# if want to use video file , simply pass the video file to this .py file
# if not using video, webcam will be used instead
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
# buffer controls the max size of deque of points
# it will maintain a buffer of x,y coordinates of object for previous 32 frames
ap.add_argument("-b", "--buffer", type=int, default=32,
                help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

# initialize the list of tracked points, 
# the frames counter and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) =(0, 0)
direction = ""

# if a video path was not supplied, grab the reference to the webcam
# this "camera" is a pointer that chooses the webcam or video file
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])
    
# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    
    # if we are viewing a video and we didi not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
    
    # resize the frame, blue it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # construct a mask for the color "green"
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    # then perform a series of dilatioins and erosions to remove any small
    # blobs left in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # find contours in the mask and initialize the current
    # (x,y) center of the ball
    # countours means outlines
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # not quite sure about the m part, need to figure that out
        
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)
            