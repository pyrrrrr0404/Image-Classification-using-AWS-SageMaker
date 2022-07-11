import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

JPEG_CONTENT_TYPE = 'image/jpeg'
JSON_CONTENT_TYPE = 'application/json'
ACCEPTED_CONTENT_TYPE = [ JPEG_CONTENT_TYPE ] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def net():
    model = models.resnet50(pretrained = True) 
    
    for param in model.parameters():
        param.requires_grad = False 
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 512), 
                             nn.ReLU(inplace = True),
                             nn.Linear(512, 256), 
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 133),
                             nn.ReLU(inplace = True)
                            )
    return model


def model_fn(model_dir):

    model = net().to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location = device))

    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):

    if content_type in ACCEPTED_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception(f"Requested an unsupported Content-Type: {content_type}, Accepted Content-Type are: {ACCEPTED_CONTENT_TYPE}")

def predict_fn(input_object, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    input_object=test_transform(input_object)
    if torch.cuda.is_available():
        input_object = input_object.cuda() 
    model.eval()
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction