#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from std_msgs.msg import Bool


def move_to_joint_positions(group, joint_positions):
    group.set_joint_value_target(joint_positions)
    plan_result = group.plan()
    # plan_result can be tuple(success, plan) or just plan depending on MoveIt version
    plan = None
    if isinstance(plan_result, tuple):
        success, plan = plan_result[0], plan_result[1]
        if not success:
            rospy.logerr("Planning failed!")
            return False
    else:
        plan = plan_result
        if not plan or len(plan.joint_trajectory.points) == 0:
            rospy.logerr("Planning failed!")
            return False

    exec_result = group.execute(plan, wait=True)
    if not exec_result:
        rospy.logerr("Execution failed!")
        return False

    group.stop()
    group.clear_pose_targets()
    return True


def control_vacuum_gripper(pub, activate=True):
    msg = Bool()
    msg.data = activate
    pub.publish(msg)
    rospy.loginfo(f"Vacuum {'ACTIVATED' if activate else 'DEACTIVATED'}")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('vacuum_pick_place_node')

    arm_group = moveit_commander.MoveGroupCommander("ur_5")
    vacuum_pub = rospy.Publisher('/vacuum_gripper/command', Bool, queue_size=1)

    rospy.sleep(2)  # Wait for connections

    rospy.loginfo("=== Starting Vacuum Pick & Place ===")

    # Joint positions (radians) for your robot - adjust if needed
    pre_grasp_joints = [0.5455, -1.0811, 1.3090, 1.3207, 1.5708, 2.1816]
    grasp_joints = [0.5774, -1.0260, 1.5337, 1.0408, 1.5808, 2.1816]
    home_joints = [0.0, -1.8325, 2.5307, 3.1415, -1.6144, 1.3090]

    # Step 1: Ensure vacuum is OFF before starting
    rospy.loginfo("Step 1: Deactivating vacuum")
    control_vacuum_gripper(vacuum_pub, activate=False)
    rospy.sleep(1.0)

    # Step 2: Move to pre-grasp pose
    rospy.loginfo("Step 2: Moving to pre-grasp pose")
    if not move_to_joint_positions(arm_group, pre_grasp_joints):
        rospy.logerr("Failed to move to pre-grasp pose")
        return

    rospy.sleep(1.0)

    # Step 3: Move to grasp pose
    rospy.loginfo("Step 3: Moving to grasp pose")
    if not move_to_joint_positions(arm_group, grasp_joints):
        rospy.logerr("Failed to move to grasp pose")
        return

    rospy.sleep(1.0)

    # Step 4: Activate vacuum gripper to pick the object
    rospy.loginfo("Step 4: Activating vacuum gripper")
    control_vacuum_gripper(vacuum_pub, activate=True)
    rospy.sleep(3.0)  # wait for suction to grip

    # Step 5: Lift object by returning to pre-grasp
    rospy.loginfo("Step 5: Lifting object")
    if not move_to_joint_positions(arm_group, pre_grasp_joints):
        rospy.logerr("Failed to lift object")
        return

    rospy.sleep(1.0)

    # Step 6: Move to home (place) pose
    rospy.loginfo("Step 6: Moving to place/home pose")
    if not move_to_joint_positions(arm_group, home_joints):
        rospy.logerr("Failed to move to home pose")
        return

    rospy.sleep(1.0)

    # Step 7: Deactivate vacuum gripper to release the object
    rospy.loginfo("Step 7: Deactivating vacuum gripper")
    control_vacuum_gripper(vacuum_pub, activate=False)
    rospy.sleep(2.0)

    rospy.loginfo("🎉 Vacuum pick & place completed successfully!")

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted")


