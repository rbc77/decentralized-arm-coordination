#!/usr/bin/env python3
import rospy
import sys
import math
import moveit_commander
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler


def generate_circular_waypoints(base_pose, radius=0.05, num_points=40, duration=3.0):
    waypoints = []
    omega = 2 * math.pi / duration
    for i in range(num_points):
        t = duration * i / num_points
        pose = Pose()
        pose.position.x = base_pose.position.x + 0.02 * math.sin(4 * 2 * math.pi * t)  # reduced sinusoidal X
        pose.position.y = base_pose.position.y + radius * math.cos(omega * t)         # smaller radius
        pose.position.z = base_pose.position.z + radius * math.sin(omega * t)
        # Option 1: Relaxed orientation (commented out fixed orientation)
        # q = quaternion_from_euler(-math.pi, 0, 0)
        # pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        # Option 2: Use current orientation from base_pose if possible
        pose.orientation = base_pose.orientation
        waypoints.append(pose)
    return waypoints


def generate_lissajous_waypoints(base_pose, A=0.04, B=0.03, num_points=40, duration=3.0):
    waypoints = []
    a = 3 * 2 * math.pi / duration
    b = 2 * 2 * math.pi / duration
    for i in range(num_points):
        t = duration * i / num_points
        pose = Pose()
        pose.position.x = base_pose.position.x + 0.02 * math.sin(4 * 2 * math.pi * t)  # reduced sinusoidal X
        pose.position.y = base_pose.position.y + A * math.sin(a * t)
        pose.position.z = base_pose.position.z + B * math.sin(b * t)
        # Use current orientation or relax fixed orientation
        pose.orientation = base_pose.orientation
        waypoints.append(pose)
    return waypoints



def execute_smooth_trajectory(group, waypoints, velocity_scaling=0.3):
    """Plan & execute a smooth trajectory from given waypoints"""
    (plan, fraction) = group.compute_cartesian_path(
        waypoints,
        0.02,                       # eef_step (2 cm resolution)
        avoid_collisions=True,      # collision checking
        path_constraints=None
    )

    if fraction < 0.9:
        rospy.logwarn(f"Only {fraction*100:.1f}% of path achievable — try smaller radius or relax orientation.")
        return False

    # Apply smoothing to velocity & acceleration
    retimed_plan = group.retime_trajectory(
        group.get_current_state(),
        plan,
        velocity_scaling_factor=velocity_scaling,
        acceleration_scaling_factor=velocity_scaling
    )

    success = group.execute(retimed_plan, wait=True)
    group.stop()
    group.clear_pose_targets()
    return success


def go_to_joint_positions(group, joint_positions):
    """Move to a specified joint configuration"""
    group.set_max_velocity_scaling_factor(1.0)
    group.set_max_acceleration_scaling_factor(1.0)
    group.set_joint_value_target(joint_positions)
    success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()
    return success


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("smooth_robot_patterns", anonymous=True)

    # Planning group for your UR5
    group = moveit_commander.MoveGroupCommander(
        "ur_5",
        robot_description="robot1/robot_description",
        ns="/robot1"
        )


    # Example home joint configuration
    home_joints = [0.0, -1.5466, 1.7882, 2.8515, -1.5949, 1.3090]

    rospy.loginfo("Moving to home joint configuration...")
    if not go_to_joint_positions(group, home_joints):
        rospy.logerr("Failed to reach home joint configuration. Aborting.")
        return
    rospy.sleep(1.0)

    base_pose = group.get_current_pose().pose

    num_cycles = 10
    cycle_duration = 0.05

    rospy.loginfo("Starting smooth Robot Dance...")

    for cycle in range(num_cycles):
        if rospy.is_shutdown():
            break
        use_circular = (cycle % 2 == 0)
        rospy.loginfo(f"Cycle {cycle+1}: {'Circular' if use_circular else 'Lissajous'} motion")

        if use_circular:
            waypoints = generate_circular_waypoints(base_pose, radius=0.1, num_points=40, duration=cycle_duration)
        else:
            waypoints = generate_lissajous_waypoints(base_pose, A=0.08, B=0.06, num_points=40, duration=cycle_duration)

        success = execute_smooth_trajectory(group, waypoints, velocity_scaling=1)
        if not success:
            rospy.logwarn(f"Cycle {cycle+1} trajectory execution failed")
        rospy.sleep(0.5)  # short break between patterns

    rospy.loginfo("Smooth Robot Dance completed!")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
