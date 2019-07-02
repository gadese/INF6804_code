# https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
# https://www.learnopencv.com/goturn-deep-learning-based-object-tracking/
# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
# https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

from __future__ import print_function
import sys
import cv2
import numpy as np
from PIL import Image
from random import randint


trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker

#Load saved images
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# CHANGE TO VIDEO OF INTEREST
# Set video to load
videoPath = "TP3_data(final)/video.mp4"
#
# # Create a video capture object to read videos
# cap = cv2.VideoCapture(videoPath)
#
# # Read first frame
# success, frame = cap.read()
# # quit if unable to read the video file
# if not success:
#     print('Failed to read video')
#     sys.exit(1)

# TEST_IMAGE_PATHS = ["TP3_data(final)/frame/" + 'frame{}.jpg'.format(i) for i in range(1, 1012)]
# image_list = []
# for image_path in TEST_IMAGE_PATHS:
#     image = Image.open(image_path)
#     image_list.append (load_image_into_numpy_array(image))

INIT_BBOX_FILE = "TP3_data(final)/init.txt"
f = open(INIT_BBOX_FILE, 'r')
box1 = f.readline()
box2 = f.readline()
box1 = box1.split(' ')[2:2+4]
box2 = box2.split(' ')[2:2+4]

bboxes = [(830, 474, 1112-830, 755-474), (1194, 433, 1479-1194, 700-433)]
colors = [(255, 0, 0), (0, 255, 0)] #Box colors


# Specify the tracker type
# trackerType = "CSRT"
for trackerType in trackerTypes:
    j = 1
    # Create a video capture object to read videos
    cap = cv2.VideoCapture(videoPath)

    # Read first frame
    success, frame = cap.read()
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)
    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    # CHANGE TO VIDEO OF INTEREST
    # Process video and track objects
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        # # # show frame
        # cv2.imshow('MultiTracker', frame)
        # # quit on ESC button
        # if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        #     break

        cv2.imwrite('results/' + trackerType + '/frame' + str(j) + '.bmp', frame)
        j += 1
