import torch
import torch.nn as nn
import threading



class iCoDeModel(nn.Module):
    def __init__(self, testra, yolo, device=torch.device("cpu")):
        super().__init__()
        self.testra = testra
        self.yolo = yolo
        self.device = device

        self.testra_output = None
        self.yolo_output = None


    def __run_testra(self, x):
        self.testra_output = None
        self.testra_output = self.testra(x)
        return self.testra(x)

    def __run_yolo(self, x):
        self.yolo_output = None
        self.yolo_output = self.yolo(x)
        return self.yolo(x)


    def forward(self, x):
        testra_thread = threading.Thread(target=self.__run_testra, args=(x,))
        yolo_thread = threading.Thread(target=self.__run_yolo, args=(x,))

        testra_thread.start()
        yolo_thread.start()

        testra_thread.join()
        yolo_thread.join()

        return self.testra_output, self.yolo_output
