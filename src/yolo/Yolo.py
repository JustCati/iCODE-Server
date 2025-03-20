import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLOWorld
from src.iou_function import iou
from src.active_objects_function import active_objects_retrieval
from statistics import mode
import sys
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "checkpoint/YOLOWorldM.pt")

class Yolo(nn.Module):
    def __init__(self, weights_path = WEIGHTS_PATH, iou_thresh = 0.1, conf_score = 0.5, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        self.weights_path = weights_path
        self.iou_thresh = iou_thresh
        self.conf_score = conf_score
        self.device = device
        self.yolo_model = YOLOWorld(self.weights_path).to(self.device)

    def get_class_vector(self, frames):
        class_vector = torch.full((1, 30), -1)
        batch_size = frames.shape[0]

        for i in range(batch_size):
            frame_np = frames[i].permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            results = self.yolo_model.predict(frame_rgb, conf=self.conf_score)
            best_obj = active_objects_retrieval(results)

            if best_obj is not None and 0 <= best_obj["obj_id"] < 25:
                class_vector[0, i] = best_obj["obj_id"]

        return class_vector

    def forward(self, x):
        class_vector = self.get_class_vector(x)
        
        class_np = class_vector.cpu().numpy()[0]

        valid_ids = class_np[class_np != -1]

        if len(valid_ids) > 0:
            return int(mode(valid_ids))
        else:
            return -1
