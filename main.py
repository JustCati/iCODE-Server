import os
import io
import sys
import time
import time
import torch
import requests
import argparse
import threading
from PIL import Image
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "testra", "src"))

from yolo.Yolo import Yolo #! FOR FUTURE USE
from testra.Testra import Testra
from iCoDeModel.iCoDeModel import iCoDeModel #! FOR FUTURE USE

from server.Server import Server
from testra.src.rekognition_online_action_detection.utils.parser import load_cfg
from testra.src.rekognition_online_action_detection.utils.env import setup_environment



# Goes into ICoDeModel
def model_worker(model, queue, frame_rate, server):
        while True:
            batch_frames = []
            for _ in range(frame_rate):
                batch_frames.append(queue.get())

            frames = []
            for frame_bytes in batch_frames:
                frame = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                frame = transforms.ToTensor()(frame)
                frames.append(frame)

            frames = torch.stack(frames, dim=0)
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                output = model(frames)

            if server.last_client_ip:
                callback_url = f"http://{server.last_client_ip}:{server.visor_callback_port}/callback/"
                payload = {
                    "timestamp": time.time(),
                    "action": int(output) #! TO CHANGE TO THE CORRECT OUTPUT
                }
                try:
                    response = requests.post(callback_url, json=payload)
                    print("Sent callback to visor at", callback_url, "Response:", response.status_code)
                except Exception as e:
                    print("Error sending callback:", e)
            else:
                print("No client IP available for callback.")




def main(args):
    FRAMERATE = 30
    cfg = load_cfg(args)
    device = setup_environment(cfg)
    cfg.MODEL.CHECKPOINT = args.testra_path #! Necessary for loading the model

    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "server", "cert", "key.pem")
    cert_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "server", "cert", "cert.pem")

    server = Server(key_file, cert_file, ip="192.168.1.101", port=34545, max_queue_size=10000)
    # model = iCoDeModel(Testra(cfg), Yolo(device), device=device) #! FOR FUTURE USE
    model = Testra(cfg)

    try:
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        model_thread = threading.Thread(target=model_worker, daemon=True, args=(model, server.get_model_queue(), FRAMERATE, server))
        model_thread.start()

        model_thread.join()
        server_thread.join()
        server.batch_queue.batch_worker_thread.join()

    except KeyboardInterrupt:
        model_thread.join()
        server_thread.join()
        server.batch_queue.batch_worker_thread.join()

        print("Server has been shut down.")
        print("Model has been shut down.")
        exit(0)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute TeSTra-Mamba on a video')
    parser.add_argument('--config_file', type=str, help='path to config file')
    parser.add_argument('--gpu', default='0', type=str, help='specify visible devices')
    parser.add_argument('opts', default=None, nargs='*', help='modify config options using the command-line',)
    parser.add_argument('--yolo_path', type=str, help='path to YOLO weights')
    parser.add_argument('--testra_path', type=str, help='path to TeSTra weights')
    args = parser.parse_args()
    main(args)
