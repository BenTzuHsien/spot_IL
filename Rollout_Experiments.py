#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from gazebo_msgs.msg import ModelStates
import math 

class Rollout:

    def __init__(self, rotation_step):
        rospy.init_node('rollout_control')
        self.rotation_step = rotation_step
        self.pub_displacement = rospy.Publisher('spot/displacement', Pose, queue_size=10, latch=True)
        self.current_yaw = 0.0
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        rospy.sleep(1) 

    def model_states_callback(self, msg):
        robot_index = msg.name.index('spot')  
        orientation = msg.pose[robot_index].orientation
        _, _, self.current_yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

    def apply_rotation_until_zero_yaw(self):
        step = 0
        while not rospy.is_shutdown():
            rospy.sleep(0.2)

            rospy.loginfo(f"Step N # {step}")
            if abs(self.current_yaw) < 0.01:
                rospy.loginfo("Yaw is approximately zero; stopping rotation.")
                break
            rospy.loginfo(f"Current Yaw is : {self.current_yaw} radians, which is {math.degrees(self.current_yaw):.2f} degrees")

            displacement = Pose()
            displacement.position = Point(0, 0, 0)
            q = quaternion_from_euler(0, 0, self.rotation_step)
            displacement.orientation = Quaternion(q[0], q[1], q[2], q[3])
            self.pub_displacement.publish(displacement)

            step += 1

if __name__ == '__main__':
    rotation_angle = -0.017   # output of our model
    rollout = Rollout(rotation_angle)
    rollout.apply_rotation_until_zero_yaw()
