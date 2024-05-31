#!/bin/bash

#SBATCH --job-name=co_sim2real
# --- SBATCH --mail-user=memmelma@cs.washington.edu ---

#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
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
export WANDB_PROJECT=openrt-baselines
export WANDB_API_KEY=638b3ee4d807a3fb5d92aca711d6288f5c3a4aeb

export MUJOCO_GL=egl
export ROOT_DIR=$(pwd)

# activate conda
eval "$(conda shell.bash hook)"
conda activate franka

srun --exclusive -N 1 -n 1 python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_30x30_co.json &
srun --exclusive -N 1 -n 1 python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_30x30_sim.json &
srun --exclusive -N 1 -n 1 python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_30x30_real.json &
srun --exclusive -N 1 -n 1 python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_isaac_sim.json &

wait

srun --exclusive -N 1 -n 1 python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_30x30_co_aug.json &
srun --exclusive -N 1 -n 1 python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_30x30_real_aug.json &
srun --exclusive -N 1 -n 1 python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_30x30_sim_aug.json &
srun --exclusive -N 1 -n 1 python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_isaac_sim_aug.json &

wait
