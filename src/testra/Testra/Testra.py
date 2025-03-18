import io
import time
import torch
import requests
from PIL import Image
import torch.nn as nn
from statistics import mode
import torchvision.transforms as transforms

from src.testra.src.rekognition_online_action_detection.models import build_model
from src.testra.src.rekognition_online_action_detection.utils.env import setup_environment
from src.testra.src.rekognition_online_action_detection.utils.checkpointer import setup_checkpointer



class Testra(nn.Module):
    def __init__(self, cfg, frame_queue, server):
        super().__init__()

        self.cfg = cfg
        self.server = server
        self.queue = frame_queue
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


    def forward(self, x):
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            x = self.backbone(x.to(self.device))
            x = x.unsqueeze(0)
            x = x.to(self.device)
            x = self.testra(x, x, x)
        return x


    def model_worker(self):
        while True:
            batch_frames = self.queue.dequeue_batch(self.cfg.MODEL.LSTR.WORK_MEMORY_LENGTH)

            frames = []
            for frame_bytes in batch_frames:
                frame = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                frame = transforms.ToTensor()(frame)
                frames.append(frame)

            frames = torch.stack(frames, dim=0)
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                output = self(frames)
            output = torch.softmax(output, dim=1)
            results = output.cpu().numpy()[0]
            max_indices = results.argmax(axis=1)
            mode_val = mode(max_indices)
            print(f"Mode: {mode_val}")

            # if mode_val != 0:
            if self.server.last_client_ip:
                callback_url = f"http://{self.server.last_client_ip}:{self.server.visor_callback_port}/callback/"
                payload = {
                    "timestamp": time.time(),
                    "action": int(mode_val)
                }
                try:
                    response = requests.post(callback_url, json=payload)
                    print("Sent callback to visor at", callback_url, "Response:", response.status_code)
                except Exception as e:
                    print("Error sending callback:", e)
            else:
                print("No client IP available for callback.")
