import cv2
import torch
from geometry_msgs.msg import Twist
import rospy

class ObstacleAvoider:
    def __init__(self):
        rospy.init_node('obstacle_avoider')
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='path_to_trained_model/best.pt')

    def avoid_obstacles(self, img):
        results = self.model(img)
        # Check if obstacles are detected
        for result in results.xyxy[0]:
            label = int(result[-1])  # Get the object label
            if label == 0:  # Assuming 'person' is class 0
                # Implement obstacle avoidance logic
                twist = Twist()
                twist.linear.x = 0.0  # Stop or adjust velocity
                self.cmd_pub.publish(twist)
                return True
        return False

    def run(self):
        cap = cv2.VideoCapture(0)  # Assuming a camera is available
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if ret:
                self.avoid_obstacles(frame)
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ObstacleAvoider().run()
