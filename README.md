# Final Project: NFL Helmet Tracking / Detetction
Final Project for Computer Vision (Fall 2021) - NFL First &amp; Future Helmet Tracking (Kaggle Challenge)

## Files
py: Contains Python scripts that execute the models in our project
- csrt_multitracker.py: Cam
- fasterrcnn_tracker.py: Cam

## Runnable Commands
- csrt_multitracker.py: The most important function is get_iou_vid(), which takes a file path to a given video and a tracker type as input. This computes the average IOU score of all predicted bounding boxes in the frames of the video, as well as displays those predictions in real time.
- fasterrcnn_tracker.py: The build_dataset() class is important for preparing the dataset of image frames and bounding boxes. Lines 165-195 execute the Fast R-CNN model training process and give an average log-loss score. The format_predictions() and plot_detected_bboxes() can be used on the validation set of images in order to display the trained model predictions on new data. 
