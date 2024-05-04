# 

# collecting data
python openrt/scripts/collect_demos_real_oculus.py exp_id=EXP_ID episodes=10 split=train language_instruction="pick up the green cube" robot.robot_type="fr3"

- exp_id: dataset name
- episode: number of episodes to collect
- language_instruction: language instruction for training, e.g., "pick up the red cube"
- split: split to collect, can either be "train" or "eval", "eval" is not necessary unless you want to check eval loss during training or run open loop evaluations on it later

stores data at: data/EXP_ID/SPLIT
hydra config file: configs/collect_demos_real.yaml


# replay collected data
python openrt/scripts/replay_demos_real.py exp_id=EXP_ID robot.blocking_control=true robot.control_hz=1 robot.robot_type="fr3"

- exp_id: experiment name, determines log directory, here also dataset name
- robot.blocking_control: whether to run blocking control
- robot.control_hz: control frequency, should be 1 if blocking control is used

replays data from: data/EXP_ID/SPLIT
stores output videos at: LOGDIR/EXP_ID
hydra config file: configs/collect_demos_real.yaml


# convert data actions -> delta (blocking)
python openrt/convert_np_to_hdf5 input_datasets=["left_25","right_25","middle_25"] output_dataset="all_75" splits=["train","eval"]

- input datasets: list of input dataset names stored at "data/NAME"
- output_dataset str of output dataset name, combined/processed dataset will be stored at "data/NAME"
- splits: splits to include in the output dataset, set to ["train"] if no "eval" sets available


default_bc_config.json
experiment.validate=true, train.hdf5_filter_key="train", train.hdf5_validation_filter_key="eval", 

# train the model
python robomimic/robomimic/scripts/train.py --name EXP_ID --config default_config.json --data data/NAME/demos.hdf5 --train.output_dir "/home/weirdlab/Projects/polymetis_franka/logdir/"

# eval model
python training/robomimic_nvidia/robomimic/scripts/eval_script.py \
    exp_id=EXP_ID robot.blocking_control=true robot.control_hz=1 robot.max_path_length=70 \
    open_loop=false data_path="data/NAME/demos.hdf5" \
    ckpt_path="path_to_model_ckpt"
     robot.robot_type="fr3"

# eval openrt
robot.blocking_control=true robot.control_hz=1


# stats = {
    #     "action": {
    #         "min": np.asarray(data["data"]["stats"]["actions_min"]),
    #         "max": np.asarray(data["data"]["stats"]["actions_max"]),
    #     }
    # }