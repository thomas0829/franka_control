

python training/robomimic_nvidia/robomimic/scripts/train.py --name robomimic_blocking_middle --config isaac_config.json --data data/green_block_sl_10_left_blocking/demos.hdf5

python training/weird_bc/eval_script.py --config-name=eval_robomimic_sim exp_id=robomimic_blocking_middle robot.camera_names=["215122255213"] robot.blocking_control=true robot.control_hz=1 robot.max_path_length=70 ckpt_path="/tmp/outputs/randomized_large/robomimic_blocking_middle/20240503173703/last.pth" open_loop=false data_path="data/green_block_sl_10_left_blocking/demos.hdf5"
