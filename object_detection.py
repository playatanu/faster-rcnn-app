import cv2
from PIL import Image

import torch
import torchvision

from coco import class_list
from helper import array_to_count

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()

def predict(image):
    color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
  
    img = Image.fromarray(color_coverted) 
    transform = torchvision.transforms.ToTensor()
    img = transform(img)

    with torch.no_grad():
        pred = model([img])

    bboxes, labels, scores = pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']
    num = torch.argwhere(scores>0.9).shape[0]

    img = color_coverted

    object_list = []

    for i in range(num):
        x1, x2, x3, x4 = bboxes[i].numpy().astype('int')
        class_name = class_list[labels.numpy()[i]-1]
        object_list.append(class_name)
        if(class_name in class_list):
            img = cv2.rectangle(img,(x1,x2),(x3,x4),(255,255,255),3)
            img = cv2.putText(img,str(class_name),(x1,x2-10),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),3)

    text = array_to_count(object_list)
    return img, text


