# generate mujoco data (single cube)
python openrt/scripts_sim/collect_demos_sim_curobo.py env=cube_5x5_fix_ori exp_id=red_cube_5x5_100 episodes=100 \
    robot.control_hz=10 robot.max_path_length=100 robot.visual_dr=false \
    language_instruction="pick up the red cube"

python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["red_cube_20x20_1k"] output_dataset="red_cube_20x20_1k" splits=["train","eval"]

python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_30x30_sim.json

training/robomimic/robomimic/logdir/tmp/mlp_red_cube_5x5_100/20240604101653/models/model_epoch_113_best_validation_0.37807576060295106.pth

python openrt/scripts/robomimic_eval.py --config-name eval_robomimic_sim \
 env=cube_20x20_fix_ori exp_id=mlp_red_cube_20x20_1k \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 \
data_path="data/red_cube_20x20_1k" \
ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/mlp_red_cube_20x20_1k/20240604103345/models/model_epoch_669_best_validation_0.171524977684021.pth" open_loop=false open_loop_split="train"
success 0.6 (w/ spatial softmax?)

python openrt/scripts/robomimic_eval.py --config-name eval_robomimic_sim \
 env=cube_20x20_fix_ori exp_id=mlp_red_cube_20x20_1k_cos_1_vis_core_basic \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 \
data_path="data/red_cube_20x20_1k" \
ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/mlp_red_cube_20x20_1k_cos_1_vis_core_basic/20240604104537/models/model_epoch_764_best_validation_0.918164336681366.pth" open_loop=false open_loop_split="train"
success 0.3

python openrt/scripts/robomimic_eval.py --config-name eval_robomimic_sim \
 env=cube_20x20_fix_ori exp_id=mlp_red_cube_20x20_1k_resnet \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 \
data_path="data/red_cube_20x20_1k" \
ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/mlp_red_cube_20x20_1k_resnet/20240604104909/models/model_epoch_837_best_validation_0.1762653648853302.pth" open_loop=false open_loop_split="train"
success 0.4

python openrt/scripts/robomimic_eval.py --config-name eval_robomimic_sim \
 env=cube_20x20_fix_ori exp_id=mlp_red_cube_20x20_1k_vis_core_basic \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 \
data_path="data/red_cube_20x20_1k" \
ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/mlp_red_cube_20x20_1k_vis_core_basic/20240604104740/models/model_epoch_931_best_validation_0.18490713238716125.pth" open_loop=false open_loop_split="train"
success 0.5

python openrt/scripts/robomimic_eval.py --config-name eval_robomimic_sim \
 env=cube_20x20_fix_ori exp_id=clip_red_cube_20x20_1k_vis_core_basic \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 \
data_path="data/red_cube_20x20_1k" \
ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/clip_frozen_30x30_sim/20240531144445/models/model_epoch_499_best_validation_-18.476616668701173.pth" open_loop=false open_loop_split="train"
success 0.



# convert mujoco data
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["red_cube_20x20_1k"] output_dataset="red_cube_20x20_1k" splits=["train","eval"]
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["red_cube_20x20_1k"] output_dataset="red_cube_20x20_1k_blocking" splits=["train","eval"]


# generate mujoco data (language multi cube)
python openrt/scripts_sim/collect_demos_sim_multi_cube_curobo.py exp_id="/media/marius/X9 Pro/pick_language_cube_500" episodes=500 \
    robot.control_hz=10 robot.max_path_length=100 robot.visual_dr=false \
    env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]"
python openrt/scripts_sim/collect_demos_sim_multi_cube_curobo.py exp_id="pick_language_cube_5" episodes=5 \
    robot.control_hz=10 robot.max_path_length=100 robot.visual_dr=false \
    env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]"
python openrt/scripts_sim/collect_demos_sim_two_cube_curobo.py exp_id="pick_language_two_cube_1k" episodes=1000 \
    robot.control_hz=10 robot.max_path_length=100 robot.visual_dr=false \
    env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]"

# convert mujoco data (language multi cube)
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["/media/marius/X9 Pro/pick_red_cube_1k_dr"] output_dataset="pick_red_cube_1k_dr_blocking" splits=["train","eval"]
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["/pick_language_cube_500"] output_dataset="pick_language_cube_500_blocking" splits=["train","eval"]
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["pick_language_cube_5"] output_dataset="pick_language_cube_5_blocking" splits=["train","eval"]
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["pick_red_cube_rnd_1k_dr"] output_dataset="pick_red_cube_rnd_1k_dr_blocking" splits=["train","eval"]
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["redcube_issac_1000_blocking_seed_10"] output_dataset="redcube_issac_1000_blocking_seed_10_blocking" splits=["train","eval"]

# replay real data
python openrt/scripts/replay_demos.py --config-name=collect_demos_sim exp_id=0518_all/0518_redblock_100_blocking robot.blocking_control=true robot.control_hz=1

# replay sim data
python openrt/scripts/replay_demos.py --config-name=collect_demos_sim exp_id=pick_red_cube_1k_blocking robot.blocking_control=true robot.control_hz=1

# analyze data
openrt/scripts/analyze_demos_hdf5.ipynb

# train
python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_config.json
python openrt/baselines/train_mt.py --config openrt/baselines/configs/diffusion_config.json
python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_lang_config.json

# eval sim
python openrt/scripts/robomimic_eval.py --config-name eval_robomimic_sim \
exp_id=co_train_sim_real \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 open_loop=false \
env.obj_pose_init="[0.5, 0.0, 0.02, 0., 0., 0., 1.]" \
data_path="data/pick_red_cube_1k_blocking" \
ckpt_path="CKPT"

# eval real
python openrt/scripts/robomimic_eval.py --config-name eval_robomimic_real \
exp_id=co_train_sim_real \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 open_loop=false \
data_path="data/pick_red_cube_1k_blocking" \
ckpt_path="CKPT"

# replay sim data
python openrt/scripts/replay_demos.py --config-name=collect_demos_sim exp_id=pick_language_cube_500_blocking robot.blocking_control=true robot.control_hz=1

# eval multi sim
python openrt/scripts_sim/robomimic_multi_eval.py --config-name eval_robomimic_sim \
exp_id=pick_language_cube_500_blocking \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 open_loop=false \
env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]" \
data_path="data/pick_language_cube_500_blocking" \
ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/mlp_lang/20240529090432/models/model_epoch_667_best_validation_0.0052105444483459.pth"

# eval multi sim (closed loop, train)
python openrt/scripts_sim/robomimic_multi_eval.py --config-name eval_robomimic_sim exp_id=pick_language_cube_500_blocking robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 open_loop=false env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]" data_path="data/pick_language_cube_500_blocking" ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/mlp_lang/20240529090432/models/model_epoch_667_best_validation_0.0052105444483459.pth" open_loop=true open_loop_split="train"

# eval multi sim (closed loop, eval)
python openrt/scripts_sim/robomimic_multi_eval.py --config-name eval_robomimic_sim exp_id=pick_language_cube_500_blocking robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 open_loop=false env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]" data_path="data/pick_language_cube_500_blocking" ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/mlp_lang/20240529090432/models/model_epoch_667_best_validation_0.0052105444483459.pth" open_loop=true open_loop_split="eval"


python openrt/scripts/robomimic_eval.py --config-name eval_robomimic_sim \
exp_id=diffusion \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 open_loop=false \
data_path="data/pick_language_one_cube_10_blocking" \
ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/diffusion/20240530160929/models/model_epoch_60.pth"

python openrt/scripts_sim/robomimic_two_cube_eval.py --config-name eval_robomimic_sim \
exp_id=rnn_pick_red_cube_rnd_1k_dr_blocking \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 open_loop=false \
data_path="data/pick_red_cube_rnd_1k_dr_blocking" \
ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/rnn_pick_language_two_cube_1k_blocking/20240531073235/models/model_epoch_984_best_validation_-31.81317138671875.pth" env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]"


python openrt/scripts/robomimic_eval.py --config-name eval_robomimic_sim \
exp_id=clip_frozen_30x30_sim \
robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 \
data_path="data/red_cube_20x20_1k" \
ckpt_path="/home/weirdlab/Projects/polymetis_franka/training/robomimic/robomimic/logdir/tmp/clip_frozen_30x30_sim/20240531144445/models/model_epoch_499_best_validation_-18.476616668701173.pth" env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]" open_loop=false open_loop_split="train"