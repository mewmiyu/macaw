import numpy as np
import cv2 as cv
from queue import Queue
import time
from threading import Thread

import utils_macaw as utils


class VideoPlayerAsync:
    """
    Class for displaying a video stream asynchronously.
    """
    def __init__(self, default_size, target_fps=30, queue_size=30):
        """
        Initializes the VideoPlayerAsync object.

        Args:
            default_size (tuple): default size of the window.
            target_fps (int, optional): target fps of the video. Defaults to 30.
            queue_size (int, optional): size of the queue. Defaults to 30.

        Returns:
            None
        """
        self.running = False
        self.Q = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.main_window, args=())
        self.thread.daemon = True
        self.fps = target_fps
        self.dt = 1.0 / self.fps
        self.default_size = np.array(default_size)
        self.window_size = self.default_size  # (height, width)
        self.ratio = float(self.window_size[0]) / float(self.window_size[1])

    def main_window(self):
        """
        Main loop of the thread.

        Returns:
            None
        """
        # create display window
        cv.namedWindow("MACAW", cv.WINDOW_KEEPRATIO)
        cv.resizeWindow("MACAW", self.window_size[0], self.window_size[1])

        t_old = time.time()
        w_old = 0
        h_old = 0
        # keep looping infinitely
        while self.running or not self.Q.empty():
            # Break if the Window is closed -> Stop the Thread and Stop the program
            if cv.getWindowProperty("MACAW", cv.WND_PROP_VISIBLE) < 1:
                break

            # Resize the Window to fit the image
            _, _, w, h = cv.getWindowImageRect("MACAW")
            if h != h_old:
                w = int(h / self.ratio)
                w_old = w
                h_old = h
            if w != w_old:
                h = int(w * self.ratio)
                h_old = h
                w_old = w
            if w != 0 and h != 0:
                cv.resizeWindow("MACAW", w, h)

            # Cap the Framerate
            t = time.time()
            elapsed = t - t_old
            t_old = t

            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)

            if self.Q.empty():
                continue

            # Render the next frame
            frame = self.Q.get()
            if w != 0 and h != 0:
                frame = utils.resize(frame.get(), w, h)

            # display the size of the queue on the frame
            cv.imshow("MACAW", frame)
            cv.waitKey(1)
        self.running = False

    def add(self, frame):
        """
        Adds a frame to the queue.

        Args:
            frame (np.ndarray): frame to be added.

        Returns:
            bool: True if the frame was added, False otherwise.
        """
        # For now manual waiting due to bugs with the player
        count = 0
        while self.Q.full() and count < 10:
            time.sleep(self.dt)
            count += 1
        if self.Q.full():
            return False
        self.Q.put(frame, timeout=10 * self.dt)
        return True

    def start(self):
        """
        Starts the thread.

        Returns:
            VideoPlayerAsync: self.
        """
        self.running = True
        self.thread.start()
        return self

    def stop(self):
        """
        Stops the thread.

        Returns:
            None
        """
        # indicate that the thread should be stopped
        self.running = False
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()
        cv.destroyAllWindows()
