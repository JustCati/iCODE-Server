import io
import cv2
import threading
import numpy as np
from PIL import Image





class Streamer():
    def __init__(self, graphic_queue):
        self.graphic_queue = graphic_queue

        self.graphic_worker_thread = threading.Thread(target=self.graphic_worker, daemon=True)
        self.graphic_worker_thread.start()


    def graphic_worker(self):
        cv2.namedWindow("FrameStream", cv2.WINDOW_NORMAL)

        while True:
            batch_frames = self.graphic_queue.get()
            for frame_data in batch_frames:
                try:
                    img_array = np.frombuffer(frame_data, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if image is None:
                        pil_img = Image.open(io.BytesIO(frame_data)).convert("RGB")
                        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    cv2.imshow("FrameStream", image)
                    if cv2.waitKey(1) & 0xFF == 27: # ESC key
                        break
                except Exception as e:
                    print("Error displaying frame:", e)

            self.graphic_queue.task_done()
