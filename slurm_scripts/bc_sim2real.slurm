#!/bin/bash

#SBATCH --job-name=sim2real
#SBATCH --mail-type=END,FAIL
# --- SBATCH --mail-user=memmelma@cs.washington.edu ---

#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=300G
#SBATCH --time=48:00:00

#SBATCH --chdir=/gscratch/weirdlab/memmelma/polymetis_franka
#SBATCH --export=all
#SBATCH --output=slurm/%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm/%j-err.txt    # where STDERR goes

# Your programs to run.
export WANDB_MODE=online
export WANDB_PROJECT=openrt
export WANDB_API_KEY=638b3ee4d807a3fb5d92aca711d6288f5c3a4aeb

export MUJOCO_GL=egl
export ROOT_DIR=$(pwd)

# activate conda
eval "$(conda shell.bash hook)"
conda activate franka

srun --exclusive -N 1 -n 1 python training/weird_bc/train_script.py exp_id=sim2real_sl_1k_img_proprio training.dataset_path=data/sim2real_sl_1k training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] training.num_epochs=100 training.eval_interval=10 &

srun --exclusive -N 1 -n 1 python training/weird_bc/train_script.py exp_id=sim2real_sl_1k_proprio training.dataset_path=data/sim2real_sl_1k training.image_keys=[] training.state_keys=["lowdim_ee","lowdim_qpos"] training.num_epochs=100 training.eval_interval=10 &

wait

srun --exclusive -N 1 -n 1 python training/weird_bc/train_script.py exp_id=sim2real_sl_1k_img training.dataset_path=data/sim2real_sl_1k training.image_keys=["front_rgb"] training.state_keys=[] training.num_epochs=100 training.eval_interval=10 &


srun --exclusive -N 1 -n 1 python training/weird_bc/train_script.py  exp_id=sim2real_ml_1k_img_proprio training.dataset_path=data/sim2real_ml_1k training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] training.num_epochs=500 training.eval_interval=10 &

wait

srun --exclusive -N 1 -n 1 python training/weird_bc/train_script.py exp_id=sim2real_ml_1k_proprio training.dataset_path=data/sim2real_ml_1k training.image_keys=[] training.state_keys=["lowdim_ee","lowdim_qpos"] training.num_epochs=500 training.eval_interval=10 &

srun --exclusive -N 1 -n 1 python training/weird_bc/train_script.py exp_id=sim2real_ml_1k_img training.dataset_path=data/sim2real_ml_1k training.image_keys=["front_rgb"] training.state_keys=[] training.num_epochs=500 training.eval_interval=10 &

wait

srun --exclusive -N 1 -n 1 python training/weird_bc/eval_script.py --config-name=bc_policy_sim exp_id=sim2real_sl_1k_img_proprio training.dataset_path=data/sim2real_sl_1k training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] robot.blocking_control=true robot.calibration_file=perception/cameras/calibration/logs/aruco/most_recent_calib.json robot.max_path_length=80 env.obj_pose_noise_dict=null &

srun --exclusive -N 1 -n 1 python training/weird_bc/eval_script.py --config-name=bc_policy_sim exp_id=sim2real_sl_1k_proprio training.dataset_path=data/sim2real_sl_1k training.image_keys=[] training.state_keys=["lowdim_ee","lowdim_qpos"] robot.blocking_control=true robot.calibration_file=perception/cameras/calibration/logs/aruco/most_recent_calib.json robot.max_path_length=80 env.obj_pose_noise_dict=null &

wait

srun --exclusive -N 1 -n 1 python training/weird_bc/eval_script.py --config-name=bc_policy_sim exp_id=sim2real_sl_1k_img training.dataset_path=data/sim2real_sl_1k training.image_keys=["front_rgb"] training.state_keys=[] robot.blocking_control=true robot.calibration_file=perception/cameras/calibration/logs/aruco/most_recent_calib.json robot.max_path_length=80 env.obj_pose_noise_dict=null &

srun --exclusive -N 1 -n 1 python training/weird_bc/eval_script.py --config-name=bc_policy_sim exp_id=sim2real_ml_1k_img_proprio training.dataset_path=data/sim2real_ml_1k training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] robot.blocking_control=true robot.calibration_file=perception/cameras/calibration/logs/aruco/most_recent_calib.json robot.max_path_length=80 &

wait

srun --exclusive -N 1 -n 1 python training/weird_bc/eval_script.py --config-name=bc_policy_sim exp_id=sim2real_ml_1k_proprio training.dataset_path=data/sim2real_ml_1k training.image_keys=[] training.state_keys=["lowdim_ee","lowdim_qpos"] robot.blocking_control=true robot.calibration_file=perception/cameras/calibration/logs/aruco/most_recent_calib.json robot.max_path_length=80 &

srun --exclusive -N 1 -n 1 python training/weird_bc/eval_script.py --config-name=bc_policy_sim exp_id=sim2real_ml_1k_img training.dataset_path=data/sim2real_ml_1k training.image_keys=["front_rgb"] training.state_keys=[] robot.blocking_control=true robot.calibration_file=perception/cameras/calibration/logs/aruco/most_recent_calib.json robot.max_path_length=80 &

wait
