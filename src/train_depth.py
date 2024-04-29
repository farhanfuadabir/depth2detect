import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from utils import *
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models, datasets, tv_tensors
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights 
from torchvision.models.detection import maskrcnn_resnet50_fpn
from transformers import pipeline

from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

##############################################################################################################

BATCH_SIZE = 2
EPOCH = 10
CKPT_DIR = "./log/ckpt"
LOG_DIR = "./log"


##############################################################################################################

transform = transforms.Compose([
    ToDepth()
])

train_dataset = datasets.CocoDetection(root="../data/coco/train2017", 
                                       annFile="../data/coco/annotations/instances_train2017.json",
                                       transform=transform)
train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset, target_keys=["boxes", "labels", "masks"])
print("\nTrain samples size:", len(train_dataset), end='\n\n')

val_dataset = datasets.CocoDetection(root="../data/coco/val2017", 
                                     annFile="../data/coco/annotations/instances_val2017.json",
                                     transform=transform)
val_dataset = datasets.wrap_dataset_for_transforms_v2(val_dataset, target_keys=["boxes", "labels", "masks"])                                     
print("\nValidation samples size:", len(val_dataset), end='\n\n')

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          collate_fn=lambda batch: tuple(zip(*batch)))

val_loader = DataLoader(dataset=val_dataset, 
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        collate_fn=lambda batch: tuple(zip(*batch)))

##############################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Available device:", device)

model_depth = models.get_model("maskrcnn_resnet50_fpn_v2", weights=None, weights_backbone=None).train()
model_depth.backbone.body.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model_depth.transform = GeneralizedRCNNTransform(800,1333,[0.485, 0.456, 0.406,0],[0.229, 0.224, 0.225,1])

##############################################################################################################

params = [p for p in model_depth.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

log_dict = {
    'train_loss': [],
    'val_loss': []
}

num_epochs = 100
model_depth = model_depth.to(device)

for epoch in range(num_epochs):
    model_depth.train()
    total_train_loss = 0
    for images, targets in train_loader:
        if 'boxes' not in targets[0].keys():
            break
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images = [image.to(device) for image in images]
        
        optimizer.zero_grad()
        loss_dict = model_depth(images, targets)

        del images
        del targets

        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        total_train_loss += losses.item()
    avg_train_loss = total_train_loss / len(train_loader)
    log_dict['train_loss'].append(avg_train_loss)
    
    model_depth.eval()
    total_val_loss = 0
    for images, targets in val_loader:
        if 'boxes' not in targets[0].keys():
            break
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images = [image.to(device) for image in images]
        # loss_dict = model_depth(images, targets)
        losses, predictions = eval_forward(model_depth, images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_val_loss += losses.item()

        del images
        del targets
        
    avg_val_loss = total_val_loss / len(val_loader)
    log_dict['val_loss'].append(avg_val_loss)

    print(f"Epoch {epoch + 1} || Train Loss: {avg_train_loss} || Val Loss: {avg_val_loss}")
    lr_scheduler.step()

    torch.save(model_depth.state_dict(), join(CKPT_DIR, f'modelDepth_{epoch}.ckpt'))
    pd.DataFrame(log_dict).to_csv(join(LOG_DIR, 'log.csv'), index=False)