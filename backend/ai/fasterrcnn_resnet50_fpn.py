import os
import shutil
from glob import glob
from pathlib import Path, PurePath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
import  xml.dom.minidom as minidom
import xmltodict
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision import datasets, models, transforms
import torchvision.models.detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import make_grid
from torchvision.ops import box_iou
from tqdm import tqdm

TRAIN_ANNOTATIONS = "./data/duomo/train/xml"
TRAIN_IMAGES = "./data/duomo/train/images"
VAL_IMAGES = "./data/duomo/valid/images"
VAL_ANNOTATIONS = "./data/duomo/valid/xml"
IMG_WIDTH = 640
EPOCHS = 1
BATCH_SIZE = 2
num_classes = 2
label_to_index = {'person': 1, 'no_object': 0}

def annotate_image(img, xml_file):
    with open(xml_file) as f:
        doc = xmltodict.parse(f.read())
    
    image = plt.imread(img)
    fig, ax = plt.subplots(1)
    ax.axis("off")
    fig.set_size_inches(10, 5)
    
    if "object" in doc.get("annotation", {}):
        annotation_objects = doc["annotation"]["object"]
        for obj in annotation_objects:
            bndbox = obj["bndbox"]
            x, y, w, h = list(map(int, obj["bndbox"].values()))
            edgecolor = {"person": "g"}
            mpatch = mpatches.Rectangle(
                (x, y),
                w-x, h-y,
                linewidth=1,
                edgecolor=edgecolor[obj["name"]],
                facecolor="none")
            ax.add_patch(mpatch)
            rx, ry = mpatch.get_xy()
            ax.annotate(obj["name"], (rx, ry),
                        color=edgecolor[obj["name"]], weight="bold",
                        fontsize=10, ha="left", va="baseline")
        ax.imshow(image)

def predict_and_annotate_image(img_path, model, device):
    # Load and transform the image
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(device).unsqueeze(0)

    # Perform prediction
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # Plotting the image and the predicted bounding boxes
    fig, ax = plt.subplots(1)
    ax.axis("off")
    ax.imshow(image.convert("RGB"))

    # Process predictions
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()

    edgecolor = {1: "g"}
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score > 0.5:  # Threshold can be adjusted
            x1, y1, x2, y2 = box
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=edgecolor.get(label, 'r'), facecolor='none')
            ax.add_patch(rect)
            ax.annotate(f'{label}: {score:.2f}', (x1, y1), color=edgecolor.get(label, 'r'), weight='bold', fontsize=10, ha='left', va='baseline')
    
    plt.show()

class CustomDataset(Dataset):
    def __init__(self, image_dir, xml_dir, transform=None):
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]  # Adjust as needed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")

        base_name = self.images[idx].split("_jpg")[0] + "_jpg"  # Retain the "_jpg" and strip the rest
        xml_file = os.path.join(self.xml_dir, f"{base_name}.xml")
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        boxes = []
        labels = []
        for obj in root.findall(".//object"):
            bndbox = obj.find('bndbox')
            x_min = int(bndbox.find('xmin').text)
            y_min = int(bndbox.find('ymin').text)
            x_max = int(bndbox.find('xmax').text)
            y_max = int(bndbox.find('ymax').text)
            boxes.append([x_min, y_min, x_max, y_max])

            class_label = label_to_index[obj.find("name").text]
            labels.append(class_label)

        if not boxes:  # No objects found
            boxes = [[0, 0, 1, 1]]  # Dummy box
            labels = [0]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        target = {'boxes': boxes, 'labels': labels}
        return image, target

def custom_collate(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return {'images': images, 'targets': targets}

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = CustomDataset(image_dir=TRAIN_IMAGES, xml_dir=TRAIN_ANNOTATIONS, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

def labels_to_tensor(labels, label_to_index, device):
    indices = [label_to_index[label] for label in labels]
    print(indices)
    return torch.tensor(indices, dtype=torch.long).to(device)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    progress_bar = tqdm(data_loader, desc="Training", leave=True)
    for batch in progress_bar:
        images = batch["images"]
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        progress_bar.set_description(f"Epoch: {epoch}, Loss: {losses.item()}")

# Move model to the GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training epochs
for epoch in range(EPOCHS):
    train_one_epoch(model, optimizer, loader, device, epoch)
    lr_scheduler.step()

torch.save(model, 'model.pth')

# Show validation image with predictions
IMG_PATH = './data/duomo/valid/images/image20240402165855_jpg.rf.8978064369cd301ced53cb654d765ced.jpg'
ANNOTATION_PATH = './data/duomo/valid/xml/image20240402165855_jpg.xml'
predict_and_annotate_image(IMG_PATH, model, device)
annotate_image(IMG_PATH, ANNOTATION_PATH)
plt.show()

def calculate_metrics(detections, ground_truths, iou_threshold):
    TP = 0
    FP = 0
    FN = 0
    iou_list = []
    for pred, true in zip(detections, ground_truths):
        pred_boxes = pred['boxes']
        true_boxes = true['boxes']
        
        if true_boxes.nelement() == 0:
            FP += pred_boxes.size(0)
            continue
        
        if pred_boxes.nelement() == 0:
            FN += true_boxes.size(0)
            continue
        
        iou_matrix = box_iou(pred_boxes, true_boxes)
        # Record IoU values
        iou_list.extend(iou_matrix.max(1).values.tolist())
        
        # IoU thresholding to determine matches
        matches = iou_matrix > iou_threshold
        
        # Count true positives, false positives, false negatives
        TP += matches.sum().item()
        FP += (matches.sum(0) == 0).sum().item()
        FN += (matches.sum(1) == 0).sum().item()

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1, iou_list

def evaluate(model, data_loader, device):
    model.eval()
    detections = []
    ground_truths = []
    all_ious = []
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            images = [image.to(device) for image in batch['images']]
            targets = [{k: v.to(device) for k, v in t.items()} if isinstance(t, dict) else {'boxes': torch.tensor([]).to(device)} for t in batch['targets']]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                detections.append({
                    'boxes': output['boxes'].cpu(),
                    'labels': output['labels'].cpu(),
                    'scores': output['scores'].cpu()
                })

                ground_truths.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })

    # Calculate metrics at different IoU thresholds
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    precisions, recalls, f1_scores = [], [], []
    for iou_threshold in iou_thresholds:
        precision, recall, f1, ious = calculate_metrics(detections, ground_truths, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        all_ious.extend(ious)

    # Plotting IoU histogram
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 2, 1)
    sns.histplot(all_ious, kde=True, bins=30)
    plt.title('Distribution of IoU Scores')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')

    # Plotting the precision
    plt.subplot(2, 2, 2)
    sns.lineplot(x=iou_thresholds, y=precisions)
    plt.title('Precision vs IoU Threshold')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Precision')

    # Plotting recall
    plt.subplot(2, 2, 3)
    sns.lineplot(x=iou_thresholds, y=recalls)
    plt.title('Recall vs IoU Threshold')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Recall')

    # Plotting F1 score
    plt.subplot(2, 2, 4)
    sns.lineplot(x=iou_thresholds, y=f1_scores)
    plt.title('F1 Score vs IoU Threshold')
    plt.xlabel('IoU Threshold')
    plt.ylabel('F1 Score')

    plt.tight_layout()
    plt.show()

val_dataset = CustomDataset(image_dir=VAL_IMAGES, xml_dir=VAL_ANNOTATIONS, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
evaluate(model, val_loader, device)
