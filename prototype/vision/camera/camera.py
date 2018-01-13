import time

import cv2


class Camera(object):
    """Camera

    Parameters
    ----------
    index : int
        Camera index

    """

    def __init__(self, index=0):
        self.frame = None
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.capture.set(cv2.CAP_PROP_EXPOSURE, 0)
        # self.capture.set(cv2.CAP_PROP_GAIN, 0)

        # Define codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        # self.out = cv2.VideoWriter('output.avi', fourcc, 120.0, (640, 480))

    def update(self):
        """Update camera

        Returns
        -------
        frame : np.array
            Camera frame

        """
        status, self.frame = self.capture.read()
        if status is False:
            raise RuntimeError("Failed to read camera frame!")

        return self.frame

    def display(self):
        """Display image frame"""
        cv2.imshow("Camera", self.frame)
        cv2.waitKey(1)

    def loop(self):
        """Loop camera"""
        frame_index = 0
        time_start = time.time()

        while True:
            self.update()
            # self.out.write(self.frame)
            self.display()
            frame_index += 1

            # print fps
            if frame_index % 100 == 0:
                time_taken = time.time() - time_start
                fps = 100.0 / time_taken
                print("fps: " + str(fps))

                time_start = time.time()
                frame_index = 0

