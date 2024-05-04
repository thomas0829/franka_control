# 

# collecting data
python openrt/scripts/collect_demos_real_oculus.py exp_id=EXP_ID episodes=10 split=train language_instruction="pick up the green cube" robot.robot_type="fr3"

- exp_id: dataset name
- episode: number of episodes to collect
- language_instruction: language instruction for training, e.g., "pick up the red cube"
- split: split to collect, can either be "train" or "eval", "eval" is not necessary unless you want to check eval loss during training or run open loop evaluations on it later

stores data at: data/EXP_ID/SPLIT
hydra config file: configs/collect_demos_real.yaml


# replay collected data (only high frequency for now - no blocking)
python openrt/scripts/replay_demos_real.py exp_id=EXP_ID robot.robot_type="fr3"

- exp_id: experiment name, determines log directory, here also dataset name

replays data from: data/EXP_ID/SPLIT
stores output videos at: LOGDIR/EXP_ID
hydra config file: configs/collect_demos_real.yaml


# convert data actions -> delta (blocking)
python openrt/scripts/convert_np_to_hdf5.py input_datasets=["EXP_ID"] output_dataset="EXP_ID_blocking" splits=["train"]

- input datasets: list of input dataset names stored at "data/NAME"
- output_dataset str of output dataset name, combined/processed dataset will be stored at "data/NAME"
- splits: splits to include in the output dataset, set to ["train"] if no "eval" sets available

example:
python openrt/scripts/convert_np_to_hdf5.py input_datasets=["left_25","right_25","middle_25"] output_dataset="all_75" splits=["train","eval"]


default_bc_config.json
experiment.validate=true, train.hdf5_filter_key="train", train.hdf5_validation_filter_key="eval", 

# train the model
python openrt/scripts/robomimic_train.py --name EXP_ID_blocking --config openrt/scripts/default_config.json --data data/EXP_ID_blocking/demos.hdf5

# eval model
python openrt/scripts/robomimic_eval.py exp_id=EXP_ID robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 open_loop=false data_path="data/EXP_ID_blocking/" ckpt_path="/home/openrt/polymetis_franka/training/robomimic_nvidia/robomimic/EXP_ID_blocking/20240504141752/last.pth" robot.robot_type="fr3"

- robomimic dumps best eval checkpoint if eval data is available -> recomended
# eval openrt
robot.blocking_control=true robot.control_hz=1
