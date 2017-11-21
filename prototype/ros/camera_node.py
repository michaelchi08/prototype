import rospy
from sensor_msgs.msg import Image

from prototype.ros.ros_node import ROSNode
from prototype.vision.camera import Camera


class CameraNode(ROSNode):
    def __init__(self):
        super(Camera, self).__init__()
        self.camera = Camera()

        self.image_topic = "/prototype/camera/image"
        self.register_publisher(self.image_topic, Image)

    def publish_image(self, frame):
        img_msg = Image()
        img_msg.data = self.camera.frame
        self.pubs[self.image_topic] = img_msg

    def loop(self):
        while not rospy.is_shutdown():
            frame = self.camera.update()
            self.publish_image(frame)


if __name__ == "__main__":
    rospy.init_node("prototype_camera_node")
