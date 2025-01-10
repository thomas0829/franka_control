import h5py
import os
import glob
from PIL import Image
import numpy as nppi
import json
import pickle
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='argument parser')  
    parser.add_argument(
        '--root_path',
        default=None,
        required=True,
        type=str)
    parser.add_argument(
        '--dataset_name',
        default=None,
        required=True,
        type=str)

    args = parser.parse_args()

    return args

def unnormalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 0.5 * (arr + 1) * (max_val - min_val) + min_val

if __name__ == "__main__":
    args = get_args()
    root_path = args.root_path
    dataset_name = args.dataset_name
    entries = []
            
    files = glob.glob(os.path.join(root_path, "demos.hdf5"))

    assert len(files) > 0, "No hdf5 files found in the root path"
    

    data = h5py.File(files[0], 'r')
    for key in data.keys():
        print(key)
    '''
    masks = data['mask']
    train_ids = []
    val_ids = []
    for t in masks['train']:
        train_ids.append(t.decode('UTF-8'))
    for t in masks['val']:
        val_ids.append(t.decode('UTF-8'))
    '''
    data = data['data']
    
    print(f'Number of trajectories: {len(data.keys())}')
    episode_cnt = 0
    stat_file = os.path.join(root_path, "stats")
    with open(stat_file, 'rb') as f:
        stats = pickle.load(f)
    stats = stats["action"]

    unseen_objects = ['carrot', 'salmon sushi', 'blue lego block', 'yellow paprika', 'peas']

    for demo_key in data.keys():
        '''
        if demo_key in train_ids:
            split = 'train'
        else:
            split = 'val'
        '''
        demo = data[demo_key]
        filename = demo_key
        normalized_actions = demo['actions']
        obs = demo['obs']
        instructions = obs['language_instruction']
        states = obs['lowdim_ee']
        proprio_keys = ["eef_pos", "eef_euler", "gripper_state"]
        images = obs['front_rgb']

        max_step = images.shape[0]

        # Iterate through the first axis of images, which represents the number of images
        print(f'Number of steps in {demo_key}: {images.shape[0]}')
        
        for i in range(states.shape[0]):
            state_dict = {}
            state_dict["eef_pos"] = states[i][:3].tolist()
            state_dict["eef_euler"] = states[i][3:6].tolist()
            state_dict["gripper_state"] = states[i][6].tolist()
            image_agentview = images[i]
            image_agentview = Image.fromarray(image_agentview)
            image_agentview.save('temp.png')
            img_dir = f"{dataset_name}/episode_{episode_cnt}/step_{i}.jpg" 
            if not os.path.exists(os.path.dirname(f'{img_dir}')):
                os.makedirs(os.path.dirname(f'{img_dir}'))
                print(f"Directory '{os.path.dirname(f'{img_dir}')}' was created.")
            image_agentview.save(img_dir)
            id_ = img_dir.replace('.jpg', '')
            #instruction = 'pick up the red block' # Temporary measure until we get instruction into the hdf5 file
            instruction = instructions[i].decode('UTF-8')
            actions = normalized_actions[i]
            actions = unnormalize(actions, stats) # unnormalizing actions for OpenRT training
            
            for unseen_object in unseen_objects:
                if unseen_object in instruction:
                    split = 'val_unseen'
            agentview_entry = {
                "id": id_,
                "image": img_dir,
                "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nWhat action should the robot take to `{instruction}`"
                },
                {
                    "from": "gpt",
                    #"value": action_values_str,
                    "raw_actions": actions.tolist(),
                    "states": state_dict
                },
                ]
            }

            entries.append(agentview_entry)
        episode_cnt+=1

    print(f'dataset size: {len(entries)}')
    with open(os.path.join(dataset_name, "task.json"), 'w') as file:
        json.dump(entries, file)