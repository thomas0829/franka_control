#!/bin/bash
# GELLO Cartesian Replay (Auto-adds _hdf5 suffix)
# Usage: ./replay_gello.sh [dataset_name]
# Example: ./replay_gello.sh gello_demo  (will use gello_demo_hdf5)

cd /home/robots/yuquand/polymetis_franka

# Get dataset name (default: gello_demo)
DATASET_NAME=${1:-gello_demo}

# Auto-construct full path with _hdf5 suffix
DATASET_PATH="/home/robots/dataset/date_114/${DATASET_NAME}_hdf5"

echo "=================================================="
echo "Replaying GELLO Cartesian Delta Demonstrations"
echo "=================================================="
echo "Dataset: $DATASET_NAME"
echo "Path: $DATASET_PATH"
echo ""
echo "⚠️  Make sure robot is in safe position!"
echo "   Press Ctrl+C to cancel within 5 seconds..."
echo ""
sleep 5

/home/robots/miniconda3/envs/polytf/bin/python openrt/scripts/replay_demos_real.py \
    +dataset_path=$DATASET_PATH \
    exp_id=replay_gello_demo \
    log.dir=/home/robots/dataset/replay_logs \
    robot=franka_real_gello \
    robot.imgs=false \
    robot.DoF=6 \
    robot.blocking_control=true \
    robot.control_hz=10

echo ""
echo "Replay complete!"
