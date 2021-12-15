# Imports

from __future__ import print_function
import sys
import cv2
from random import randint
import pandas as pd
import numpy as np
import os
import subprocess
from IPython.display import Video, display

# My working directory for the data, change it to fit your computer

working_dir = "/Users/camrogers/Downloads/Computer Vision/NFL Kaggle Data/"

# This is the video labels for each frame with the players and bounding boxes

vid_labels = pd.read_csv(working_dir + "train_labels.csv")

# Types of tracking algorithms that could be used

def create_tracker_csrt():
  tracker = cv2.legacy_TrackerCSRT.create()
  return tracker


# IOU score function

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
  
# Function to train CSRT tracker on a video, compute IOU score, and display video with tracker's predictions

def get_iou_vid(videoPath, trackerType):
    # Create a video capture object to read videos

    cap = cv2.VideoCapture(working_dir+"/train/" + videoPath)

    # Read first frame

    success, frame = cap.read()

    # quit if unable to read the video file

    if not success:
      print('Failed to read video')
      sys.exit(1)

    ## List for boxes, colors of boxes, and player labels

    bboxes = []
    colors = []
    labels = []

    # First frame may be a black screen, this loop goes to the next frame until it's not black

    while True:
        if np.max(frame) == 0:
            success, frame = cap.read()
            if not success:
                print('Failed to read video')
                sys.exit(1)
        else:
            break

    # Bounding boxes of first frame of the given video

    bbox_df = vid_labels.query("video == @videoPath & frame == 1")

    for i in range(bbox_df.shape[0]):
        bboxes.append([bbox_df.iloc[i, 6], bbox_df.iloc[i, 8], bbox_df.iloc[i, 7], bbox_df.iloc[i, 9]])
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        labels.append(bbox_df.iloc[i, 5])
    
    print('Selected bounding boxes {}'.format(bboxes)) 

    # Create MultiTracker object

    multiTracker = cv2.legacy_MultiTracker.create()

    # Initialize MultiTracker

    for bbox in bboxes:
      multiTracker.add(create_tracker_csrt(), frame, bbox)

    # j is a counter representing the current frame that the video is on

    j = 1

    # List of iou scores for each frame/player bounding box

    iou_scores = []

    # Updating the tracker

    while cap.isOpened():
      success, frame = cap.read()
      if not success:
          break
      # get updated location of objects in subsequent frames
      success, boxes = multiTracker.update(frame)
      j += 1
      # draw tracked objects
      for i, newbox in enumerate(boxes):
        # label of particular player helmet
    
        lab = labels[i]
    
        # predicted bounding boxes
    
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

        # drawing predictions on video
    
        #cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        #cv2.putText(frame, labels[i], (p1[0], max(0, p2[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=colors[i], thickness=2)

        # getting true box labels
    
        bb_true = vid_labels.query("video == @videoPath & frame == @j & label == @lab")
    
        # player may drop out of video and bounding box no longer exists, just skip that box
    
        if bb_true.shape[0] == 0:
          continue
    
        # true bounding box from train_labels
    
        p1_true = (int(bb_true.iloc[0, 6]), int(bb_true.iloc[0, 8]))
        p2_true = (int(bb_true.iloc[0, 6]+bb_true.iloc[0, 7]), int(bb_true.iloc[0, 8]+bb_true.iloc[0, 9]))

        # IOU score for particular box/frame
    
        iou = bb_intersection_over_union([p1_true[0], p1_true[1], p2_true[0], p2_true[1]],
                                     [p1[0], p1[1], p2[0], p2[1]])
        iou_scores.append(iou)
    

      # show frame
      cv2.imshow('MultiTracker', frame)
      # quit on ESC button
      if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
          break

    cv2.destroyAllWindows()

    # Average IOU score for video

    print(np.mean(iou_scores))
    
    return np.mean(iou_scores)


# Taking random subset of 15 videos to test our tracker on (only used 15 because of runtime concerns)

video_names_end = vid_labels["video"].str.contains("Endzone")
video_names = vid_labels["video"][video_names_end].unique()

rand_idx = np.random.randint(0, len(video_names), 15)

video_names = video_names[rand_idx]

iou_scores = []

for name in video_names:
  iou = get_iou_vid(name, "CSRT")
  iou_scores.append(iou)
  print(iou)

np.mean(iou_scores) # About 0.42

