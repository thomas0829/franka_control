import os
import glob
import pickle
import argparse
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_image_file(file_path):
    # Load the image file
    image = Image.open(file_path)
    # Rotate the image 180 degrees
    image = image.rotate(180)
    # Crop the image to 720 by 720
    width, height = image.size
    new_width = 720
    left = (width - new_width) / 2
    top = 0
    right = (width + new_width) / 2
    bottom = height
    image = image.crop((left, top, right, bottom))
    # Convert the image to a numpy array
    return np.array(image)

def matrix_to_6dof(transform_matrix):
    transform_matrix = np.array(transform_matrix)
    x, y, z = transform_matrix[:3, 3]
    rotation = R.from_matrix(transform_matrix[:3, :3])
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
    return x, y, z, roll, pitch, yaw

def process_file(file_path, file_type):
    if file_type == 'pickle':
        return load_pickle_file(file_path)
    elif file_type == 'image':
        return load_image_file(file_path)

def process_files(subdir):
    data_dict = {
        'action_left': [],
        'action_right': [],
        'color': [],
        'lowdim_ee': [],
        'language_instruction': []
    }
    action_files = glob.glob(os.path.join(subdir+'/action', '*.pkl'))
    image_files = glob.glob(os.path.join(subdir+'/camera', 'color_*'))
    action_files.sort()
    image_files.sort()
    for i in range(len(action_files)):
        action_data = process_file(action_files[i], 'pickle')
        data_dict['action_left'].append(np.concatenate([np.array(matrix_to_6dof(action_data['left_goal_pose'])), [action_data['left_gripper_command']]]))
        data_dict['action_right'].append(np.concatenate([np.array(matrix_to_6dof(action_data['right_goal_pose'])), [action_data['right_gripper_command']]]))
        data_dict['color'].append(process_file(image_files[i], 'image'))

    # Add dummy lowdim_ee data (you may need to replace this with actual data)
    data_dict['lowdim_ee'] = np.zeros((len(data_dict['action_left']), 6))

    # Add dummy language instruction (you may need to replace this with actual data)
    data_dict['language_instruction'] = ['pick up the red block'] * len(data_dict['action_left'])

    return data_dict

def convert_to_numpy_array(data_dict):
    return np.array([(key, np.array(value)) for key, value in data_dict.items()],
                    dtype=object)

def process_trajectories(main_dir, output_dir):
    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)
        if os.path.isdir(subdir_path):
            print(f'Processing: {subdir_path}')
            data_dict = process_files(subdir_path)
            #data_array = convert_to_numpy_array(data_dict)
            data_array = np.array([data_dict])
            
            output_file = os.path.join(output_dir, f"{subdir}.npy")
            np.save(output_file, data_array)
            print(f"Saved data for {subdir} to {output_file}")
            
def main():
    parser = argparse.ArgumentParser(description="Process rummy data and save as a single .npy file per trajectory")
    parser.add_argument('--main_dir', type=str, required=True, help="Main directory containing the rummy trajectories")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the .npy files")

    args = parser.parse_args()
    main_dir = args.main_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_trajectories(main_dir, output_dir)
    print(f"All data processed and saved to {output_dir}")

if __name__ == "__main__":
    main()