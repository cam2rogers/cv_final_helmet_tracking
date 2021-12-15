### Code Adapted From Shayan Taherian Kaggle Submission: https://www.kaggle.com/shayantaherian/nfl-training-fasterrcnn

# Import Packages

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import imageio

from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm 
from tqdm.notebook import tqdm as tqdm

import cv2
import os
import re

import random
import subprocess

from PIL import Image
from IPython.display import Video, display

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

import ast

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

# Build Dataset of Image Frames

class build_dataset():
    # Below is the file path for my folder of images, but change it to fit your computer
    def __init__(self):
        self.images_df = pd.read_csv('image_labels.csv')
        self.images_list = os.listdir(r'/home/jupyter-rogersdv/images/images')
        # Classes of helmets to be tracked
        self.labels_dict = {'Helmet': 1,
                           'Helmet-Blurred': 2,
                           'Helmet-Difficult': 3,
                           'Helmet-Sideline': 4,
                           'Helmet-Partial': 5}
    
    # Collect initial bounding boxes of images
    def __getitem__(self, idx):
        img_path = os.path.join(r'/home/jupyter-rogersdv/images/images', self.images_list[idx])
        img = np.array(Image.open(img_path)) / 255
        img = np.moveaxis(img, 2, 0) # to [C, H, W]
        
        # Collect data about boxes and helmet labels from `image_labels.csv`
        img_data_df = self.images_df[self.images_df['image'] == self.images_list[idx]]     
        n_bboxes = img_data_df.shape[0]
        bboxes = []
        labels = []
        for i in range(n_bboxes):
            img_data = img_data_df.iloc[i]
            x_min = img_data.left
            x_max = img_data.left + img_data.width
            y_min = img_data.top
            y_max = img_data.top + img_data.height
            bboxes.append([x_min, y_min, x_max, y_max])
            label = self.labels_dict[img_data.label]
            labels.append(label)
         
        # Convert data to tensors
        img = torch.as_tensor(img, dtype=torch.float32)    
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        
        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        target['image_id'] = image_id
        
        return img, target
    
    def __len__(self):
        return len(self.images_list)
      
      
  
 # Functions to convert train/validation image information to tensors
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# Loading Fast R-CNN (ResNet50 Model), pretrained weights on COCO dataset
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier by adjusting number of classes
num_classes = 6  
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# Function to form mini-batches of tensors
def collate_fn(batch):
    return tuple(zip(*batch))

# Building dataset and making train/validation split

dataset = build_dataset()

indices = torch.randperm(len(dataset)).tolist()
train_set = int(0.9*len(indices))

train_dataset = torch.utils.data.Subset(dataset, indices[:train_set])
valid_dataset = torch.utils.data.Subset(dataset, indices[train_set:])

train_data_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size = 8,
                shuffle = False,
                collate_fn = collate_fn)

valid_data_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size = 8,
                shuffle = False,
                collate_fn = collate_fn)

# Allocating model to a device, creating a model optimizer using SGD

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = None

# Model training & optimization

num_iters = 100


progress_bar = tqdm(range(num_iters))
tr_it = iter(train_data_loader)
loss_log = []
iterations = []

# Looping through batches and copmuting log loss
for i in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_data_loader)
        data = next(tr_it)
    model.train()
    torch.set_grad_enabled(True)
    imgs, targets = data
    imgs = [image.to(device) for image in imgs]
    targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]
    loss_dict = model(imgs, targets)
    losses = sum(loss for loss in loss_dict.values())
    
    # Optimization
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
        
    loss_log.append(losses.item())
    iterations.append(i)
    # Updating a live progress bar
    progress_bar.set_description(f'batch loss: {losses.item()}, average loss: {np.mean(loss_log)}.')
    
    
# Plotting log loss performance by iteration
plt.plot(iterations, loss_log)
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Average Log Loss per Batch")
plt.show()

# Function to put image bounding boxes and detection scores in string format
def format_predictions(boxes, scores):
    pred_strings = []
    for s, b in zip(scores, boxes.astype(int)):
        pred_strings.append(f'{s:.4f} {b[0]} {b[1]} {b[2] - b[0]} {b[3] - b[1]}')

    return " ".join(pred_strings)

# Evaluating model and compiling successfully detected bounding boxes
detection_threshold = 0.5
results = []
device = 'cuda'
model.eval()
for images, image_ids in valid_data_loader:

    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        result = {
            'image_id': image_id,
            'PredictionString': format_predictions(boxes, scores)
        }

        
        results.append(result)
        
# Function to plot predicted bounding boxes

def plot_detected_bboxes(test_img, predictions, n_to_plot=2, score_threshold=0.6):
    
    n = min(len(test_img), n_to_plot)
    
    fig, ax = plt.subplots(1, n, figsize=(20, 8))
    
    for i in range(n):
        img = np.asarray(test_img[i].cpu().numpy() * 255, dtype=np.int64)
        img = np.moveaxis(img, 0, 2)
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        ax[i].imshow(img)
        ax[i].set_axis_off()

        bboxes = predictions[i]['boxes'].cpu().numpy()
        scores = predictions[i]['scores'].cpu().numpy()
        scores_mask = scores > score_threshold
        for bbox in bboxes[scores_mask]:
            patch = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=2,
                edgecolor='r',
                facecolor='None',
                alpha=0.8)
            ax[i].add_patch(patch)  
        
    fig.tight_layout()
    return 
  
# Evaluate / Plot validation set images
  
model.eval()
torch.set_grad_enabled(False)

test_it = iter(valid_data_loader)

valid_img, valid_gt  = next(test_it)
valid_img = [image.to(device) for image in valid_img]

predictions = model(test_img)

plot_detected_bboxes(valid_img, predictions,
                     n_to_plot=2,
                     score_threshold=0.6)
