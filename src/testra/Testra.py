import torch
import torch.nn as nn

from src.testra.src.rekognition_online_action_detection.models import build_model
from src.testra.src.rekognition_online_action_detection.utils.env import setup_environment
from src.testra.src.rekognition_online_action_detection.utils.checkpointer import setup_checkpointer



class Testra(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = setup_environment(cfg)
        checkpointer = setup_checkpointer(cfg, phase='test')

        # Build backbone
        effnet = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=True)
        self.backbone = nn.Sequential(*list(effnet.children())[:-1], 
                                *list(list(effnet.children())[-1].children())[:-2],
                                nn.AvgPool1d(kernel_size=257, stride=1))
        self.backbone.to(self.device)
        self.backbone.eval()

        # Build testra
        self.testra = build_model(cfg, self.device)
        self.testra.to(self.device)
        self.testra.eval()

        checkpointer.load(self.testra)


    #TODO: include the mode calculation in here instead of outside
    def forward(self, x):
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            x = self.backbone(x.to(self.device))
            x = x.unsqueeze(0)
            x = x.to(self.device)
            x = self.testra(x, x, x)
        return x
