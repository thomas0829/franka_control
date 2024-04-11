python openrt/scripts/collect_demos_sim_curobo.py exp_id=synth_sl_100 act_noise_std=0.005 episodes=100 env.obj_pose_noise_dict=null

# diffusion
# # train
python training/weird_diffusion/train_script.py training.dataset_path="data/synth_sl_100/train" training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=diffusion_synth_sl_100_img_proprio

python training/weird_diffusion/eval_script.py training.dataset_path="data/synth_sl_100/train" training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=diffusion_synth_sl_100_img_proprio env.obj_pose_noise_dict=null

python training/weird_diffusion/train_script.py training.dataset_path="data/synth_sl_100/train" training.image_keys=["front_rgb"] training.state_keys=[] training.with_state=false exp_id=diffusion_synth_sl_100_img

python training/weird_diffusion/eval_script.py training.dataset_path="data/synth_sl_100/train" training.image_keys=["front_rgb"] training.state_keys=[] training.with_state=false exp_id=diffusion_synth_sl_100_img env.obj_pose_noise_dict=null

# BC

python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_img

python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_img env.obj_pose_noise_dict=null

python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_100_img_proprio

python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_100_img_proprio env.obj_pose_noise_dict=null


python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_front_wrist

python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["wrist_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_100_wrist_proprio
python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_100_front_proprio
python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["wrist_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_wrist
python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_front
################################################################
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["wrist_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_100_wrist_proprio
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_100_front_proprio
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["shoulder_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_100_shoulder_proprio
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["wrist_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_wrist
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_front
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["shoulder_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_shoulder

CUDA_VISIBLE_DEVICES=1 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb","shoulder_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_front_shoulder
CUDA_VISIBLE_DEVICES=1 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["front_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_front_wrist
CUDA_VISIBLE_DEVICES=1 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["shoulder_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_shoulder_wrist
CUDA_VISIBLE_DEVICES=1 python training/basic_bc/train_script.py training.dataset_path="data/synth_sl_100" training.image_keys=["shoulder_rgb","wrist_rgb","front_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_shoulder_wrist_front
############################

CUDA_VISIBLE_DEVICES=0 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100"   env.obj_pose_noise_dict=null training.image_keys=["wrist_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_100_wrist_proprio
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100"   env.obj_pose_noise_dict=null training.image_keys=["front_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_100_front_proprio
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100"   env.obj_pose_noise_dict=null training.image_keys=["shoulder_rgb"] training.state_keys=["lowdim_ee","lowdim_qpos"] exp_id=bc_synth_sl_100_shoulder_proprio
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100"   env.obj_pose_noise_dict=null training.image_keys=["wrist_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_wrist
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100"   env.obj_pose_noise_dict=null training.image_keys=["front_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_front
CUDA_VISIBLE_DEVICES=0 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100"   env.obj_pose_noise_dict=null training.image_keys=["shoulder_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_shoulder

CUDA_VISIBLE_DEVICES=1 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100" env.obj_pose_noise_dict=null training.image_keys=["front_rgb","shoulder_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_front_shoulder
CUDA_VISIBLE_DEVICES=1 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100" env.obj_pose_noise_dict=null training.image_keys=["front_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_front_wrist
CUDA_VISIBLE_DEVICES=1 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100" env.obj_pose_noise_dict=null training.image_keys=["shoulder_rgb","wrist_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_shoulder_wrist
CUDA_VISIBLE_DEVICES=1 python training/basic_bc/eval_script.py training.dataset_path="data/synth_sl_100" env.obj_pose_noise_dict=null training.image_keys=["shoulder_rgb","wrist_rgb","front_rgb"] training.state_keys=[] exp_id=bc_synth_sl_100_shoulder_wrist_front
############################