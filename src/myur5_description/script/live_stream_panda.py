#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class PandaCameraViewer:
    def __init__(self):
        rospy.init_node('panda_camera_viewer', anonymous=True)

        self.image_topic = "/depth_camera/image_raw"  # updated topic
        self.bridge = CvBridge()
        self.frame = None

        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.loginfo("Panda camera live stream started. Press 'q' to quit.")

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)

    def run(self):
        cv2.namedWindow("Panda Camera Feed", cv2.WINDOW_NORMAL)
        rate = rospy.Rate(30)  # 30 FPS display loop
        while not rospy.is_shutdown():
            if self.frame is not None:
                cv2.imshow("Panda Camera Feed", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            rate.sleep()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    viewer = PandaCameraViewer()
    viewer.run()
