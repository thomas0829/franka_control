#!/bin/bash
# GELLO Data Collection
# Usage: ./record_gello.sh [exp_id] [episodes] [start_traj]

cd /home/duanj1/thomas/franka_control

# Fix dataset directory permissions if needed
if [ -d "/home/duanj1/dataset" ]; then
    sudo chown -R robots:robots /home/duanj1/dataset 2>/dev/null || true
fi

# Fix USB device permissions
sudo chmod 666 /dev/ttyUSB0 2>/dev/null || true

/home/duanj1/anaconda3/envs/franka/bin/python openrt/scripts/collect_demos_gello.py \
    robot=franka_real_gello \
    robot.imgs=false \
    exp_id=${1:-gello_demo} \
    episodes=${2:-3} \
    start_traj=${3:-0} \
    +rec=true \
    +max_duration=60 \
    +instruction="Put the cube into the plate." \
    hydra.run.dir=/home/duanj1/thomas/franka_control/openrt/scripts/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}
