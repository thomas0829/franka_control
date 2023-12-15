import tensorflow as tf
import numpy as np
from tf_agents.policies import py_tf_eager_policy
import tf_agents
from tf_agents.trajectories import time_step as ts
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow_hub as hub


def as_gif(images):
    # Render the images as the gif:
    images[0].save(
        "/tmp/temp.gif", save_all=True, append_images=images[1:], duration=1000, loop=0
    )
    gif_bytes = open("/tmp/temp.gif", "rb").read()
    return gif_bytes

def embed_task_str(msg, embed):

    # episode_natural_language_instruction = steps[0][rlds.OBSERVATION]['natural_language_instruction'].numpy().decode()

    def normalize_task_name(task_name):
        replaced = task_name.replace('_', ' ').replace('1f', ' ').replace(
            '4f', ' ').replace('-', ' ').replace('50',
                                                ' ').replace('55',
                                                                ' ').replace('56', ' ')
        return replaced.lstrip(' ').rstrip(' ')

    return embed([normalize_task_name(msg)])[0]


# Load TF model checkpoint
# Replace saved_model_path with path to the parent folder of
# the folder rt_1_x_tf_trained_for_002272480_step.
saved_model_path = "models/rt_1_x_tf_trained_for_002272480_step"

tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    model_path=saved_model_path, load_specs_from_pbtxt=True, use_tf_function=True
)

# Obtain a dummy observation, where the features are all 0
observation = tf_agents.specs.zero_spec_nest(
    tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation)
)

# msg = "go towards the red block and pick up the red block"
color = "blue"
msg = f"go towards the {color} block and pick up the {color} block"
embed = hub.load(
    'https://tfhub.dev/google/universal-sentence-encoder-large/5')

embedding = embed_task_str(msg, embed)

#Add language embedding to observation
observation["natural_language_embedding"] = embedding

del embed

# # Construct a tf_agents time_step from the dummy observation
# tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))
# # Initialize the state of the policy
# policy_state = tfa_policy.get_initial_state(batch_size=1)
# # Run inference using the policy
# action = tfa_policy.action(tfa_time_step, policy_state)

def resize(image):
    image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
    image = tf.cast(image, tf.uint8)
    return image


policy_state = tfa_policy.get_initial_state(batch_size=1)

predicted_actions = []
images = []

horizon = 500

from robot.robot_env import RobotEnv

env = RobotEnv(
    hz=10,
    DoF=3,
    robot_model="panda",
    ip_address="172.16.0.1", # "172.16.0.1",
    camera_ids=[0],
    camera_model="zed",
    max_lin_vel=0.2,
    max_rot_vel=1.0,
    max_path_length=horizon,
)

obs = env.reset()
img = obs["img_obs_0"]
imgs = [img]
acts = []

for step in range(horizon):
    image = resize(obs["img_obs_0"])

    images.append(image)
    observation["image"] = image

    tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))

    policy_step = tfa_policy.action(tfa_time_step, policy_state)
    action = policy_step.action
    policy_state = policy_step.state

    # prepare action
    pos_delta = action["world_vector"]
    rot_delta = action["rotation_delta"]
    gripper_delta = action["gripper_closedness_action"]
    act = np.append(action["world_vector"], action["gripper_closedness_action"])
    # execute on robot
    obs, reward, done, _ = env.step(act)

    predicted_actions.append(action)


action_name_to_values_over_time = defaultdict(list)
predicted_action_name_to_values_over_time = defaultdict(list)
figure_layout = [
    "terminate_episode_0",
    "terminate_episode_1",
    "terminate_episode_2",
    "world_vector_0",
    "world_vector_1",
    "world_vector_2",
    "rotation_delta_0",
    "rotation_delta_1",
    "rotation_delta_2",
    "gripper_closedness_action_0",
]
action_order = [
    "terminate_episode",
    "world_vector",
    "rotation_delta",
    "gripper_closedness_action",
]

for i, action in enumerate(predicted_actions):
    for action_name in action_order:
        for action_sub_dimension in range(action[action_name].shape[0]):
            # print(action_name, action_sub_dimension)
            title = f"{action_name}_{action_sub_dimension}"

            predicted_action_name_to_values_over_time[title].append(
                predicted_actions[i][action_name][action_sub_dimension]
            )

figure_layout = [["image"] * len(figure_layout), figure_layout]

plt.rcParams.update({"font.size": 12})

stacked = tf.concat(tf.unstack(images[::3], axis=0), 1)

fig, axs = plt.subplot_mosaic(figure_layout)
fig.set_size_inches([45, 10])

for i, (k, v) in enumerate(predicted_action_name_to_values_over_time.items()):
    axs[k].plot(v, label="predicted action")
    axs[k].set_title(k)
    axs[k].set_xlabel("Time in one episode")

axs["image"].imshow(stacked.numpy())
axs["image"].set_xlabel("Time in one episode (subsampled)")

plt.legend()

plt.savefig("test.png")
