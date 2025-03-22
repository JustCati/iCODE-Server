import torch
import numpy as np
import torch.nn as nn
from statistics import mode
from torchvision.models import efficientnet_v2_s

from src.testra.src.rekognition_online_action_detection.models import build_model
from src.testra.src.rekognition_online_action_detection.utils.env import setup_environment
from src.testra.src.rekognition_online_action_detection.utils.checkpointer import setup_checkpointer



class Testra(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = setup_environment(cfg)
        checkpointer = setup_checkpointer(cfg, phase='test')

        # Percentage of prediction different from the background (0) 
        # to calculate the mode of the predictions: 30%
        self.threshold = float(cfg.MODEL.LSTR.WORK_MEMORY_LENGTH) * 30.0 / 100

        # Build backbone
        self.backbone = efficientnet_v2_s(weights="DEFAULT")
        if "distilled" in cfg.INPUT.VISUAL_FEATURE:
            self.backbone.classifier = nn.Linear(1280, 1024)
        else:
            self.backbone.classifier = nn.Identity()
        self.backbone.to(self.device)
        self.backbone.eval()

        # Build testra
        self.testra = build_model(cfg, self.device)
        self.testra.to(self.device)
        self.testra.eval()

        checkpointer.load(self.testra)


    def forward(self, x):
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            x = self.backbone(x.to(self.device))
            x = x.unsqueeze(0)
            x = x.to(self.device)
            x = self.testra(x, x, x)

        # Mode calculation
        output = torch.softmax(x, dim=1)
        results = output.cpu().numpy()[0]
        max_indices = results.argmax(axis=1)

        idx_nonzero = np.where(max_indices != 0)[0]
        if idx_nonzero.shape[0] <= self.threshold:
            return 0

        return mode(max_indices[idx_nonzero])
