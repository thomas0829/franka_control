#!/bin/bash

#SBATCH --job-name=bc_diff_sanity
#SBATCH --mail-type=END,FAIL
# --- SBATCH --mail-user=memmelma@cs.washington.edu ---

#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=100G # 300G
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

# # train
# srun --exclusive -N 1 -n 1 python training/weird_diffusion/train_script.py training.dataset_path="data/synth_sl/train" training.image_keys=["left_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=diffusion_synth_sl_img_proprio &
# srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl/train" training.image_keys=["left_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_img_proprio &
# srun --exclusive -N 1 -n 1 python training/weird_diffusion/train_script.py training.dataset_path="data/synth_sl/train" training.image_keys=["left_rgb"] training.state_keys=[] training.with_state=false exp_id=diffusion_synth_sl_img &
# srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl/train" training.image_keys=["left_rgb"] training.state_keys=[] exp_id=bc_synth_sl_img &
# srun --exclusive -N 1 -n 1 python training/weird_diffusion/train_script.py training.dataset_path="data/green_block/train" training.image_keys=["215122255213_rgb"] training.state_keys=[] training.with_state=false exp_id=diffusion_green_block_img &
# srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/green_block/train" training.image_keys=["215122255213_rgb"] training.state_keys=[] exp_id=bc_green_block_img &
# srun --exclusive -N 1 -n 1 python training/weird_diffusion/train_script.py training.dataset_path="data/green_block/train" training.image_keys=["215122255213_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=diffusion_green_block_img_proprio &
# srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/green_block/train" training.image_keys=["215122255213_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_green_block_img_proprio &

# wait

# srun --exclusive -N 1 -n 1 python training/weird_diffusion/train_script.py training.dataset_path="data/synth_1k/train" training.image_keys=["left_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=diffusion_synth_1k_img_proprio &
# srun --exclusive -N 1 -n 1 python training/weird_diffusion/train_script.py training.dataset_path="data/synth_1k/train" training.image_keys=["left_rgb"] training.state_keys=[] training.with_state=false exp_id=diffusion_synth_1k_img &
# srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/synth_1k/train" training.image_keys=["left_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_1k_img_proprio &
# srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/synth_1k/train" training.image_keys=["left_rgb"] training.state_keys=[] exp_id=bc_synth_1k_img &

# wait

# eval
srun --exclusive -N 1 -n 1 python training/weird_diffusion/eval_script.py training.dataset_path="data/synth_sl/train" training.image_keys=["left_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=diffusion_synth_sl_img_proprio &
srun --exclusive -N 1 -n 1 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl/train" training.image_keys=["left_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_img_proprio &
srun --exclusive -N 1 -n 1 python training/weird_diffusion/eval_script.py training.dataset_path="data/synth_sl/train" training.image_keys=["left_rgb"] training.state_keys=[] training.with_state=false exp_id=diffusion_synth_sl_img &
srun --exclusive -N 1 -n 1 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl/train" training.image_keys=["left_rgb"] training.state_keys=[] exp_id=bc_synth_sl_img &
srun --exclusive -N 1 -n 1 python training/weird_diffusion/eval_script.py training.dataset_path="data/green_block/train" training.image_keys=["215122255213_rgb"] training.state_keys=[] training.with_state=false exp_id=diffusion_green_block_img &
srun --exclusive -N 1 -n 1 python training/basic_bc/eval_script.py training.dataset_path="data/green_block/train" training.image_keys=["215122255213_rgb"] training.state_keys=[] exp_id=bc_green_block_img &
wait
srun --exclusive -N 1 -n 1 python training/weird_diffusion/eval_script.py training.dataset_path="data/green_block/train" training.image_keys=["215122255213_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=diffusion_green_block_img_proprio &
srun --exclusive -N 1 -n 1 python training/basic_bc/eval_script.py training.dataset_path="data/green_block/train" training.image_keys=["215122255213_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_green_block_img_proprio &

srun --exclusive -N 1 -n 1 python training/weird_diffusion/eval_script.py training.dataset_path="data/synth_1k/train" training.image_keys=["left_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=diffusion_synth_1k_img_proprio &
srun --exclusive -N 1 -n 1 python training/weird_diffusion/eval_script.py training.dataset_path="data/synth_1k/train" training.image_keys=["left_rgb"] training.state_keys=[] training.with_state=false exp_id=diffusion_synth_1k_img &
# srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/synth_1k/train" training.image_keys=["left_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_1k_img_proprio &
# srun --exclusive -N 1 -n 1 python training/basic_bc/train_script.py training.dataset_path="data/synth_1k/train" training.image_keys=["left_rgb"] training.state_keys=[] exp_id=bc_synth_1k_img &

wait