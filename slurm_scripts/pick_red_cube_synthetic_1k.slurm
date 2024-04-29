#!/bin/bash

#SBATCH --job-name=explore_multi_param
#SBATCH --mail-type=END,FAIL
# --- SBATCH --mail-user=memmelma@cs.washington.edu ---

#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=100G
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

export EXP_ID=diffusion_policy_pick_red_cube_synthetic_1k
export DATA_ID=pick_red_cube_synthetic_1k
# activate conda
eval "$(conda shell.bash hook)"
conda activate franka

# collect demos
# srun --exclusive -N 1 -n 1 python $ROOT_DIR/openrt/scripts/collect_demos_sim_curobo.py exp_id=$DATA_ID episodes=1000

# convert data to RLDS
# cd $ROOT_DIR/convert/weird_rlds_dataset_builder/$DATA_ID
# srun --exclusive -N 1 -n 1 tfds build --overwrite --data_dir $ROOT_DIR/data/
# cd $ROOT_DIR

# train diffusion policy
srun --exclusive -N 1 -n 1  python $ROOT_DIR/training/weird_diffusion/train_script.py exp_id=$EXP_ID training.dataset_path=$ROOT_DIR/data/$DATA_ID/train

# eval diffusion policy
srun --exclusive -N 1 -n 1  python $ROOT_DIR/training/weird_diffusion/eval_script.py exp_id=$EXP_ID
