import torch
import torch.nn as nn
import threading



class iCoDeModel(nn.Module):
    def __init__(self, testra, yolo, device=torch.device("cpu")):
        super().__init__()
        self.testra = testra
        self.yolo = yolo
        self.device = device


    def __run_testra(self, x):
        return self.testra(x)

    def __run_yolo(self, x):
        return self.yolo(x)


    def forward(self, x):
        testra_output = [None]
        yolo_output = [None]

        testra_thread = threading.Thread(target=self.__run_testra, args=(x,))
        yolo_thread = threading.Thread(target=self.__run_yolo, args=(x,))

        testra_thread.start()
        yolo_thread.start()

        testra_thread.join()
        yolo_thread.join()

        #TODO: logics here that checks when testra is = 1 then look for yolo
        #TODO: or just sends everything and then the visor do what it wants??

        return testra_output, yolo_output #* testra_output = mode of the testra output, yolo_output = active object (how we manage the mode vs frame by frame prediction of yolo?)
