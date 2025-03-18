import torch.nn as nn


#! This is a dummy class to represent the YOLO model.
class Yolo(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x
