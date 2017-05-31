import cv2


class Camera(object):
    def __init__(self, index=0):
        self.frame = None
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    def update(self):
        status, self.frame = self.capture.read()

        if status is False:
            raise RuntimeError("Failed to read camera frame!")

        return self.frame

    def display(self):
        cv2.imshow("Camera", self.frame)
        cv2.waitKey(1)

    def loop(self):
        while True:
            self.update()
            self.display()
