### EE JOINT POS ### -> works!
env._reset_joint_qpos = np.array([0.0, -0.8, 0.0, -2.3, 0.1, 2.3, 0.8])
obs = env.reset()

file = file_names[0]
episode = h5py.File(file, "r+")["data"]["demo_0"]

poss = []
eulers = []

from utils.transformations import *
for i in range(50):
    env._robot.update_joints((episode["obs"]["joint_pos"][i]).tolist(), velocity=False, blocking=True)
    poss.append(env._robot.get_ee_pos())
    eulers.append(env._robot.get_ee_angle())
    

### EE POSE ### -> looks like it works! verufy ...
env._reset_joint_qpos = np.array([0.0, -0.8, 0.0, -2.3, 0.1, 2.3, 0.8])
env.reset()
file_idx=0
file = file_names[file_idx]
episode = h5py.File(file, "r+")["data"]["demo_0"]

from utils.transformations import *
from utils.transformations_mujoco import *

# OFFSET FROM WORLD FRAME TO ROBOT FRAME
world_offset_pos = np.array([0.2045, 0., 0.])
ee_offset_euler = np.array([0., 0., -np.pi / 4])

# BLOCKING CONTROL BY RUNNING WITH LOWER CONTROL FREQUENCY Hz
env.control_hz = 1

first_grasp_idx = np.where(episode["actions"][..., -1] == -1)[0][0]
actions = episode["actions"][:].copy()
actions[first_grasp_idx:] = 1

jointss = []
poss = []
poss_des = []
eulers = []
eulers_des = []
imgs = []


for i in range(len(episode["obs"]["eef_pos"])):

    start_time = time.time()
    
    desired_ee_pos = episode["obs"]["eef_pos"][i] + world_offset_pos
    desired_ee_quat = episode["obs"]["eef_quat"][i]
    desired_ee_euler = quat_to_euler(desired_ee_quat)
    desired_ee_euler = add_angles(ee_offset_euler, desired_ee_euler)

    gripper = actions[i,-1]

    env._update_robot(
                np.concatenate((desired_ee_pos, desired_ee_euler, [gripper])),
                action_space="cartesian_position",
                blocking=False,
            )

    jointss.append(env._robot.get_joint_positions())

    poss.append(env._robot.get_ee_pos())
    poss_des.append(desired_ee_pos)
    eulers.append(env._robot.get_ee_angle())
    eulers_des.append(desired_ee_euler)

    imgs.append(env.render())

    # SLEEP TO MAINTAIN CONTROL FREQUENCY
    comp_time = time.time() - start_time
    sleep_left = max(0, (1 / env.control_hz) - comp_time)
    time.sleep(sleep_left)

import imageio
imageio.mimsave(f"ee_pose_{file_idx}.gif", np.stack(imgs), duration=5.)

plt.close()
poss = np.stack(poss)
poss_des = np.stack(poss_des)
plt.plot(poss[...,0], color="tab:orange", label="x real")
plt.plot(episode["obs"]["eef_pos"][...,0], color="tab:orange", linestyle="dotted", label="x isaac")
#plt.plot(poss_des[...,0], color="tab:orange", linestyle="--", label="x real (commanded)")
plt.plot(poss[...,1], color="tab:blue", label="y real")
plt.plot(episode["obs"]["eef_pos"][...,1], color="tab:blue", linestyle="dotted", label="y isaac")
#plt.plot(poss_des[...,1], color="tab:blue", linestyle="--", label="y real (commanded)")
plt.plot(poss[...,2], color="tab:pink", label="z real")
plt.plot(episode["obs"]["eef_pos"][...,2], color="tab:pink", linestyle="dotted", label="z isaac")
#plt.plot(poss_des[...,2], color="tab:pink", linestyle="--", label="z real (commanded)")
plt.legend()
plt.savefig(f"ee_pos_{file_idx}.png")

plt.close()
eulers = np.stack(eulers)
eulers_des = np.stack(eulers_des)
plt.plot(eulers[...,0], color="tab:orange", label="roll real")
plt.plot(quat_to_euler(episode["obs"]["eef_quat"])[...,0], color="tab:orange", linestyle="dotted", label="roll isaac")
#plt.plot(eulers_des[...,0], color="tab:orange", linestyle="--", label="roll real (commanded)")
plt.plot(eulers[...,1], color="tab:blue", label="pitch real")
plt.plot(quat_to_euler(episode["obs"]["eef_quat"])[...,1], color="tab:blue", linestyle="dotted", label="pitch isaac")
#plt.plot(eulers_des[...,1], color="tab:blue", linestyle="--", label="pitch real (commanded)")
plt.plot(eulers[...,2], color="tab:pink", label="yaw real")
plt.plot(quat_to_euler(episode["obs"]["eef_quat"])[...,2], color="tab:pink", linestyle="dotted", label="yaw isaac")
#plt.plot(eulers_des[...,2], color="tab:pink", linestyle="--", label="yaw real (commanded)")
plt.legend()
plt.savefig(f"ee_angles_{file_idx}.png")

plt.close()
jointss = np.stack(jointss)
colors = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
for j in range(7):
    if j == 0:
        plt.plot(jointss[...,j], label=f"joint {j} real", color=colors[j])
        plt.plot(episode["obs"]["joint_pos"][..., j], label=f"joint {j} isaac", color=colors[j], linestyle="dashed")
    else:
        plt.plot(jointss[...,j], color=colors[j])
        plt.plot(episode["obs"]["joint_pos"][..., j], color=colors[j], linestyle="dashed")
plt.legend()
plt.savefig(f"joint_pos_{file_idx}.png")