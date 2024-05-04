# passwords
## FR3 login (web)
abhgupta@cs.washington.edu
pw: WeirdLab123$

## FR3 NUC
ssh nuc@172.16.0.1
pw: robot

conda activate polymetis-fr3
### launch gripper server
launch_gripper.py gripper=franka_hand
### launch robot server
launch_robot.py robot_client=franka_hardware
### bridge both to the laptop
python  polymetis_franka/robot/real/run_server.py

## FR3 laptop
openrt
pw: weirdl@123 

conda activate polymetis_franka


# collecting data
python openrt/scripts/collect_demos_real_oculus.py exp_id=EXP_ID episodes=10 split=train language_instruction="pick up the green cube" robot.robot_type="fr3" robot.control_hz=15

- exp_id: dataset name
- episode: number of episodes to collect
- language_instruction: language instruction for training, e.g., "pick up the red cube"
- split: split to collect, can either be "train" or "eval", "eval" is not necessary unless you want to check eval loss during training or run open loop evaluations on it later
- robot.robot_type: robot type (fr3, panda) to make sure correct IK is chosen

stores data in .npy format at: data/EXP_ID/SPLIT
hydra config file: configs/collect_demos_real.yaml


# replay collected data (only high frequency for now - no blocking)
python openrt/scripts/replay_demos_real.py exp_id=EXP_ID robot.robot_type="fr3"

- exp_id: experiment / dataset name, determines log directory

replays data from: data/EXP_ID/SPLIT
stores output videos at: LOGDIR/EXP_ID
hydra config file: configs/collect_demos_real.yaml


# convert data actions -> delta (blocking)
python openrt/scripts/convert_np_to_hdf5.py input_datasets=["EXP_ID"] output_dataset="EXP_ID_blocking" splits=["train"]

- input datasets: list of input dataset names stored at "data/NAME"
- output_dataset str of output dataset name, combined/processed dataset will be stored at "data/NAME"
- splits: splits to include in the output dataset, set to ["train"] if no "eval" sets available

hydra config file: configs/convert_demos_real.yaml

additional example w/ multiple datasets:
python openrt/scripts/convert_np_to_hdf5.py input_datasets=["left_25","right_25","middle_25"] output_dataset="all_75" splits=["train","eval"]
 

# train the model
do export WANDB_API = API_KEY if you want to log wandb & modify openrt/scripts/default_config.json to include your wandb project/run_name etc.

python openrt/scripts/robomimic_train.py --name EXP_ID_blocking --config openrt/scripts/default_config.json --data data/EXP_ID_blocking/demos.hdf5

--name : experiment name
--config : robomimic config file
--data : path to demos.hdf5 data file (includes "demo.hdf5")

visualizing eval loss:
- set the following commands in openrt/scripts/default_config.json : experiment.validate=true, train.hdf5_filter_key="train", train.hdf5_validation_filter_key="eval",


# eval model
python openrt/scripts/robomimic_eval.py exp_id=EXP_ID robot.blocking_control=true robot.control_hz=1 robot.max_path_length=100 open_loop=false data_path="data/EXP_ID_blocking/" ckpt_path="/home/openrt/polymetis_franka/training/robomimic_nvidia/robomimic/EXP_ID_blocking/20240504141752/last.pth" robot.robot_type="fr3"

- exp_id: experiment name
- robot.blocking_control: to run blocking control
- robot.control_hz: set to 1 Hz when running blocking control. 
- robot.max_path_length: number of steps to run eval for, try to set to a number close to the traj you've collected (video will be dumped at the end!)
- open_loop: support for running open loop predictions on the training / eval data
- data_path: path to path to demos.hdf5 data file (doesn't include "demo.hdf5"), used to load normalization stats and open loop data
- ckpt_path: path to model checkpoint, full path recommended as robomimic stores models in weird directories..., try to use best eval checkpoint
- robot.robot_type: robot type (fr3, panda) to make sure correct IK is chosen


# eval openrt
python openrt/scripts/openrt_eval.py exp_id=EXP_ID robot.blocking_control=true robot.control_hz=1 robot.robot_type="fr3"

- exp_id: experiment name
- robot.blocking_control: to run blocking control
- robot.control_hz: set to 1 Hz when running blocking control
- robot.robot_type: robot type (fr3, panda) to make sure correct IK is chosen
