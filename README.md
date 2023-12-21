# software setup

## control
install mamba from miniforge
mamba env create -f environment.yaml

## oculus reader
https://github.com/rail-berkeley/oculus_reader

## realsense cameras
pip install pyrealsense2

## zed cameras
SDK from https://www.stereolabs.com/developers/release

# hardware setup
https://docs.google.com/document/d/1ag-32UJUf0TF95TZb7whSicRCi6NJ9ER18cc9D_EeiI/edit?usp=sharing

# examples
python scripts/collect_demos.py --exp pick_red --dof 6 --max_episode_length 1000 --episodes 25
python scripts/behavior_cloning.py --exp pick_red --mode train --modality state --hidden_dim 128 --epochs 100000 --batch_size 32 --lr 3e-4

# todos
- gripper on FR3, panda (WEIRD), panda (RSE) seem to behave differently...