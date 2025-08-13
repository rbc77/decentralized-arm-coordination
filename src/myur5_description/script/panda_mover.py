#!/usr/bin/env python3

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

rospy.init_node("robot2_panda_simple_mover")

# Action client to the namespaced trajectory controller
client = actionlib.SimpleActionClient(
    '/robot2/panda_controller/follow_joint_trajectory',
    FollowJointTrajectoryAction
)

rospy.loginfo("Waiting for /robot2 trajectory controller...")
client.wait_for_server()

# Create the trajectory goal
goal = FollowJointTrajectoryGoal()
goal.trajectory.joint_names = [
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7'
]

# Example joint target
point = JointTrajectoryPoint()
point.positions = [0.3, -1.57, 0, -1.57, 3, 1.3, 0]  # radians
point.time_from_start = rospy.Duration(3.0)

goal.trajectory.points.append(point)

# Send and wait
rospy.loginfo("Sending goal to robot2...")
client.send_goal(goal)
client.wait_for_result()
rospy.loginfo("Motion complete!")
