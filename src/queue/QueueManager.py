import queue
import threading


class QueueManager():
    def __init__(self, frame_queue, batch_size=30, maxsize=10000):
        self.maxsize = maxsize
        self.batch_size = batch_size

        self.frame_queue = frame_queue
        self.model_queue = queue.Queue(maxsize=self.maxsize)
        self.graphic_queue = queue.Queue(maxsize=self.maxsize)

        self.dequeue_manager_thread = threading.Thread(target=self.dequeue_manager, daemon=True)
        self.dequeue_manager_thread.start()


    def dequeue_manager(self):
        while True:
            batch_frames = self.frame_queue.dequeue_batch(self.batch_size)

            self.model_queue.put(batch_frames)
            self.graphic_queue.put(batch_frames, timeout=0.1)
