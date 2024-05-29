# generate mujoco data (single cube)
python openrt/scripts_sim/collect_demos_sim_curobo.py exp_id=pick_red_cube_rnd_1k_dr episodes=1000 \
    robot.control_hz=10 robot.max_path_length=100 robot.visual_dr=true \
    env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]" \
    language_instruction="pick up the red cube"

# convert mujoco data
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["pick_red_cube_1k"] output_dataset="pick_red_cube_1k_blocking" splits=["train","eval"]
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["pick_red_cube_rnd_1k_dr"] output_dataset="pick_red_cube_rnd_1k_dr_blocking" splits=["train","eval"]


# generate mujoco data (language multi cube)
python openrt/scripts_sim/collect_demos_sim_multi_cube_curobo.py exp_id="/media/marius/X9 Pro/pick_language_cube_500" episodes=500 \
    robot.control_hz=10 robot.max_path_length=100 robot.visual_dr=false \
    env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]"
python openrt/scripts_sim/collect_demos_sim_multi_cube_curobo.py exp_id="pick_language_cube_5" episodes=5 \
    robot.control_hz=10 robot.max_path_length=100 robot.visual_dr=false \
    env.obj_pose_init="[0.45, 0.0, 0.02, 0., 0., 0., 1.]"

# convert mujoco data (language multi cube)
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["/media/marius/X9 Pro/pick_red_cube_1k_dr"] output_dataset="pick_red_cube_1k_dr_blocking" splits=["train","eval"]
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["/pick_language_cube_500"] output_dataset="pick_language_cube_500_blocking" splits=["train","eval"]
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["pick_language_cube_5"] output_dataset="pick_language_cube_5_blocking" splits=["train","eval"]
python openrt/scripts/convert_np_to_hdf5.py --config-name=convert_demos_sim input_datasets=["pick_red_cube_1k_dr"] output_dataset="pick_red_cube_1k_dr_blocking" splits=["train","eval"]

# replay real data
python openrt/scripts/replay_demos.py --config-name=collect_demos_sim exp_id=0518_all/0518_redblock_100_blocking robot.blocking_control=true robot.control_hz=1

# replay sim data
python openrt/scripts/replay_demos.py --config-name=collect_demos_sim exp_id=pick_red_cube_1k_blocking robot.blocking_control=true robot.control_hz=1

# analyze data
openrt/scripts/analyze_demos_hdf5.ipynb

# train
python openrt/baselines/train_mt.py --config openrt/baselines/configs/mlp_config.json
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