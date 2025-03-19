import torch
import numpy as np
import cv2
from ultralytics import YOLOWorld
from iou_function import iou    
from active_objects_function import active_objects_retrieval

WEIGHTS_PATH = "/storage/icode/iCODE-Server/src/yolo/checkpoint/YOLOWorldM.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IOU_THRESH = 0.1
CONF_SCORE = 0.5

def get_yolo_classes(frames):

    yolo_model = YOLOWorld(WEIGHTS_PATH).to(device)

    class_vector = torch.full((1, 30), -1)
    print("class vector", class_vector)
    
    batch_size = frames.shape[0]
    # Forse non è meglio creare una batch di 30 immagini e fare 
    # una sola inferenza anziché 30 inferenze su singole immagini?
    for i in range(batch_size):
        if i >= 30:
            break
            
        frame_np = frames[i].permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        # Run YOLOWorld prediction
        results = yolo_model.predict(frame_rgb, conf=CONF_SCORE)
        
        # Retrieve the active object
        best_obj = active_objects_retrieval(results)
        
        if best_obj is not None and 0 <= best_obj["obj_id"] < 25:  
            class_vector[0, i] = best_obj["obj_id"]
    
    return class_vector
