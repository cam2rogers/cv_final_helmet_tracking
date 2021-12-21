# Final Project: NFL Helmet Tracking / Detetction
Final Project for Computer Vision (Fall 2021) - NFL First &amp; Future Helmet Tracking [(Kaggle Challenge)](https://www.kaggle.com/c/nfl-impact-detection/overview)
Data descriptions and download can be found [here](https://www.kaggle.com/c/nfl-impact-detection/data)

## Files
py: Contains Python scripts that execute the models in our project
- csrt_multitracker.py: Cam
- fastrcnn_tracker.py: Cam, Kelly
- yolov3_model.py: Aidan

## Runnable Commands and File Descriptions
- csrt_multitracker.py: The most important function is get_iou_vid(), which takes a file path to a given video and a tracker type as input. This computes the average IOU score of all predicted bounding boxes in the frames of the video, as well as displays those predictions in real time.
- fasterrcnn_tracker.py: The build_dataset() class is important for preparing the dataset of image frames and bounding boxes. Lines 165-195 execute the Faster R-CNN model training process and give an average log-loss score. The format_predictions() and plot_detected_bboxes() can be used on the validation set of images in order to display the trained model predictions on new data. 
- yolov3_model.py: This file loads in the weights file and the configuration file for the yolov3 model and then attempts to detect football players in images. In the py folder, there is a folder titled “Images” which contains four images labeled sample0.jpg, sample1.jpg, sample2.jpg, sample3jpg, and sample4jpg. This folder also contains the results after using the model to perform football player detection on these images and the results are labeled output_sample0.jpg, output_sample1.jpg, output_sample2.jpg, output_sample3.jpg, output_sample4.jpg.
