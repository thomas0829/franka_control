# setup
conda create -n polymetis python=3.8
conda install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis
pip install gym==0.22.0
pip install pyrealsense2
pip install opencv-python