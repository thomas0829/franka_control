####### REPLAY JOINT POS

env._reset_joint_qpos = np.array([0.0, -0.8, 0.0, -2.3, 0.1, 2.3, 0.8])
obs = env.reset()

file = file_names[0]
episode = h5py.File(file, "r+")["data"]["demo_0"]

from utils.transformations import *

for i in range(30):
    env._robot.update_joints((episode["obs"]["joint_pos"][i]).tolist(), velocity=False, blocking=True)

##################


####### REPLAY EE POSE (polymetis IK)

env._reset_joint_qpos = np.array([0.0, -0.8, 0.0, -2.3, 0.1, 2.3, 0.8])
env.reset()
file = file_names[3]
episode = h5py.File(file, "r+")["data"]["demo_0"]

# OFFSET FROM WORLD FRAME TO ROBOT FRAME
world_offset = np.array([0.2045, 0., 0.])

for i in range(30):
    desired_ee_pos = episode["obs"]["eef_pos"][i] + world_offset
    # 45 DEGREE OFFSET MISSING IN EE SPACE
    desired_ee_quat = episode["obs"]["eef_quat"][i]
    env._robot.move_to_ee_pose(desired_ee_pos.tolist(), desired_ee_quat.tolist())

##################


####### REPLAY EE POSE (R2D2 IK)
# -> works in SIM!
# -> real has offset issue -> maybe an additional 45 degrees from offset sim, i.e., +np.pi/2 ?

from scipy.spatial.transform import Rotation as R

env._reset_joint_qpos = np.array([0.0, -0.8, 0.0, -2.3, 0.1, 2.3, 0.8])
env.reset()
file = file_names[0]
episode = h5py.File(file, "r+")["data"]["demo_0"]
from utils.transformations import *

# OFFSET FROM WORLD FRAME TO ROBOT FRAME
world_offset_pos = np.array([0.2045, 0., 0.])
# 45 DEGREE OFFSET IN EE SPACE
ee_offset_euler = np.array([0., 0., np.pi / 4])

# BLOCKING CONTROL BY RUNNING WITH LOWER CONTROL FREQUENCY Hz
env.control_hz = 1

first_grasp_idx = np.where(episode["actions"][..., -1] == -1)[0][0]
actions = episode["actions"][:].copy()
actions[first_grasp_idx:] = 1

imgs = []

for i in range(len(actions)):

    start_time = time.time()
    
    desired_ee_pos = episode["obs"]["eef_pos"][i] + world_offset_pos
    desired_ee_quat = episode["obs"]["eef_quat"][i]

    desired_ee_euler = quat_to_euler(desired_ee_quat) + ee_offset_euler

    gripper = actions[i,-1]

    env._update_robot(
                np.concatenate((desired_ee_pos, desired_ee_euler, [gripper])),
                action_space="cartesian_position",
                blocking=False,
            )

    # SLEEP TO MAINTAIN CONTROL FREQUENCY
    comp_time = time.time() - start_time
    sleep_left = max(0, (1 / env.control_hz) - comp_time)
    time.sleep(sleep_left)

##################


####### CHECK IK

env._reset_joint_qpos = np.array([0.0, -0.8, 0.0, -2.3, 0.1, 2.3, 0.8])
env.reset()
file = file_names[3]
episode = h5py.File(file, "r+")["data"]["demo_0"]

from utils.transformations import *

# from robot.real.inverse_kinematics.robot_ik_solver import RobotIKSolver
# env._robot._ik_solver._ik_solver = RobotIKSolver(
#             robot_type="panda", control_hz=10, SPEED=True
#         )

i = 10

desired_ee_pos = episode["obs"]["eef_pos"][i]
desired_ee_quat = episode["obs"]["eef_quat"][i]

cartesian_delta_pos = desired_ee_pos - obs["lowdim_ee"][:3] + np.array([0.2045, 0., 0.])
cartesian_delta_angle = quat_to_euler(desired_ee_quat) - obs["lowdim_ee"][3:6]
cartesian_delta = np.concatenate((cartesian_delta_pos, cartesian_delta_angle))

robot_state = env._robot.get_robot_state()[0]
joint_delta = env._robot._ik_solver.cartesian_velocity_to_joint_velocity(cartesian_delta, robot_state)
qdes = obs["lowdim_qpos"][:7] + joint_delta

##################
