#!/bin/bash

#SBATCH --job-name=bc_1k_multi
#SBATCH --mail-type=END,FAIL
# --- SBATCH --mail-user=memmelma@cs.washington.edu ---

#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=300G
#SBATCH --time=24:00:00

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

srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/synth_1k" training.image_keys=["front_rgb","shoulder_rgb"] training.state_keys=[] exp_id=bc_synth_1k_front_shoulder &
srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/synth_1k" training.image_keys=["front_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_1k_front_wrist &
srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/synth_1k" training.image_keys=["shoulder_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_1k_shoulder_wrist &
srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/synth_1k" training.image_keys=["front_rgb","shoulder_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_1k_front_shoulder_wrist &

wait

srun --exclusive -N 1 -n 1 python training/basic_bc/eval_script.py training.dataset_path="data/synth_1k" training.image_keys=["front_rgb","shoulder_rgb"] training.state_keys=[] exp_id=bc_synth_1k_front_shoulder &
srun --exclusive -N 1 -n 1 python training/basic_bc/eval_script.py training.dataset_path="data/synth_1k" training.image_keys=["front_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_1k_front_wrist &
srun --exclusive -N 1 -n 1 python training/basic_bc/eval_script.py training.dataset_path="data/synth_1k" training.image_keys=["shoulder_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_1k_shoulder_wrist &
srun --exclusive -N 1 -n 1 python training/basic_bc/eval_script.py training.dataset_path="data/synth_1k" training.image_keys=["front_rgb","shoulder_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_1k_front_shoulder_wrist &

wait