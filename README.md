```markdown
# Dual-Robot Gazebo Demo – UR5 & Franka Emika Panda

> A ROS/Gazebo simulation in which a **UR5** performs a choreographed “robot-dance” while a **Franka Emika Panda** on the opposite table **visually tracks** the green object on the UR5 gripper and **mimics its motion** in real time using Image-Based Visual Servoing (IBVS).

---

## 🎬 Demo Video  
*(https://github.com/rbc77/decentralized-arm-coordination/blob/main/sample_working_video%20(1).mp4)*

---

## 📦 Repository Structure
```
```text
myur5_description/
├── launch/
│   └── myur5.launch               # spawns both robots + tables
├── config/
│   ├── ur5_moveit_config/         # MoveIt! setup for UR5
│   └── panda/                     # camera plugin, URDF patches
├── scripts/
│   ├── motion.py                  # UR5 dance trajectory
│   ├── panda_mover.py             # Panda “home” pose
│   ├── ik_solver.py               # green-object tracker (legacy)
│   └── panda_mimic.py             # IBVS mimicking node
└── README.md                      # this file
---

## 🚀 Quick Start

1. **Clone & build**
   ```bash
   cd ~/catkin_ws/src
   git clone <this-repo>
   cd ..
   catkin_make
   source devel/setup.bash
   ```

2. **Launch the full scene**
   ```bash
   # Terminal 1 – Gazebo world + both robots
   roslaunch myur5_description myur5.launch

   # Terminal 2 – MoveIt! for UR5
   roslaunch ur5_moveit_config moveit_planning_execution.launch

   # Terminal 3 – Panda initial pose
   rosrun myur5_description panda_mover.py

   # Terminal 4 – green-object tracker
   rosrun myur5_description ik_solver.py

   # Terminal 5 – IBVS mimicking
   rosrun myur5_description panda_mimic.py

   # Terminal 6 – UR5 dance routine
   rosrun myur5_description motion.py
   ```

---

## 🤖 Features in Detail

| Component | What it does | Key Techniques |
|-----------|--------------|----------------|
| **UR5** | Performs a smooth 10-cycle dance alternating **circular**, **Lissajous** and **sinusoidal** Cartesian trajectories | MoveIt!, `move_group`, time-parameterized Cartesian paths |
| **Panda** | 1. Looks at UR5’s green target via end-effector RGB camera.<br>2. **Tracks** the object with HSV thresholding + multi-tracker fallback (CSRT→KCF→MOSSE) + Kalman filter.<br>3. **Mimics** the target motion using Image-Based Visual Servoing (IBVS) while respecting joint limits. | OpenCV, `cv_bridge`, IBVS, null-space redundancy, clamping |
| **Gazebo World** | Two tables, two robots, no collision between robot arms | Namespaced URDFs |

---

## 🔧 Technical Notes

### Spawning Two Robots
* Each robot is pushed into its **namespace** (`robot1/`, `robot2/`) to avoid TF and `robot_description` clashes.
* Tables are modeled as simple **static boxes** inside each URDF for convenience.

### Pick & Place (skipped)
* Original plan: use Roboteq fingers → vacuum gripper → Gazebo vacuum plugin.  
* Due to time constraints, a **green cube is rigidly attached** to the UR5 gripper to simulate “picked” state.

### IBVS Architecture
```
Camera image
     ↓  HSV threshold
Contours → Score → Tracker → Kalman-smoothed centroid
     ↓  Motion gate (≤3 px / 0.2 s)
IBVS controller (e → L → v_cam → v_base → J⁺ → q̇)
     ↓
JointTrajectory → /robot2/arm_controller/joint_trajectory
```

---

## 📋 Dependencies

* Ubuntu 20.04 / ROS Noetic
* Gazebo 11
* MoveIt!
* OpenCV & `cv_bridge`
* `franka_ros`, `universal_robot` 

Install extras:
```bash
sudo apt install ros-noetic-moveit ros-noetic-franka-ros ros-noetic-universal-robots
```

---

## 🛠️ Tuning & Debugging

| Topic | Purpose |
|-------|---------|
| `/robot2/camera/image_raw` | raw RGB from Panda cam |
| `/robot2/green_target/centroid` | tracked object (u,v,Z) |
| `/robot2/green_target/quality` | 0–1 confidence score |
| `/move_group/monitored_planning_scene` | MoveIt! collision scene |

HSV sliders window can be enabled by setting `enable_hsv_trackbars:=true` in `panda_mimic.py`.

---

## 📝 TODO / Contributions Welcome

- [ ] Fix vacuum gripper plugin for real pick & place  
- [ ] Add dynamic reconfigure for IBVS gains  
- [ ] Replace HSV tracker with deep-learning detector (e.g., YOLO)  
- [ ] RViz panel for on-the-fly trajectory switching

---
