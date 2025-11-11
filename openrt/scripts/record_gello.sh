#!/bin/bash
# GELLO Data Collection
# Usage: ./record_gello.sh [exp_id] [episodes] [start_traj]

cd /home/robots/yuquand/polymetis_franka

# Fix dataset directory permissions if needed
if [ -d "/home/robots/dataset" ]; then
    sudo chown -R robots:robots /home/robots/dataset 2>/dev/null || true
fi

# Fix USB device permissions
sudo chmod 666 /dev/ttyUSB0 2>/dev/null || true

/home/robots/miniconda3/envs/polytf/bin/python openrt/scripts/collect_demos_gello.py \
    robot=franka_real_gello \
    robot.imgs=false \
    exp_id=${1:-gello_demo} \
    episodes=${2:-1} \
    start_traj=${3:-0} \
    hydra.run.dir=/home/robots/yuquand/polymetis_franka/openrt/scripts/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}
