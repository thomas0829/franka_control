#!/usr/bin/env python3
"""
簡單的 pickle 轉 HDF5 腳本
"""
import os
import sys
import pickle
import h5py
import numpy as np
from glob import glob

# 從命令行參數讀取路徑
if len(sys.argv) < 2:
    print("Usage: python simple_convert_to_hdf5.py <pickle_dir> [output_dir]")
    print("Example: python simple_convert_to_hdf5.py /home/robots/dataset/date_1031/m&m_pickup_pickle")
    sys.exit(1)

pickle_dir = sys.argv[1]
if len(sys.argv) >= 3:
    output_dir = sys.argv[2]
else:
    # 自動生成 output_dir（把 _pickle 換成 _hdf5）
    if pickle_dir.endswith('_pickle'):
        output_dir = pickle_dir.replace('_pickle', '_hdf5')
    else:
        output_dir = pickle_dir + '_hdf5'

os.makedirs(output_dir, exist_ok=True)

# 讀取所有 pickle 檔案
pickle_files = sorted(glob(os.path.join(pickle_dir, "*.pkl")))
print(f"找到 {len(pickle_files)} 個 pickle 檔案")

# 建立 HDF5
hdf5_path = os.path.join(output_dir, "demos.hdf5")
f = h5py.File(hdf5_path, "w")
grp = f.create_group("data")

# 收集所有 actions 用於統計
all_actions = []

# 轉換每個 episode
for i, pkl_file in enumerate(pickle_files):
    print(f"處理 {os.path.basename(pkl_file)}...")
    
    with open(pkl_file, 'rb') as pf:
        data = pickle.load(pf)
    
    # 建立 episode group
    ep_name = f"demo_{i}"
    ep_grp = grp.create_group(ep_name)
    
    # 儲存 actions
    actions = data['action']
    ep_grp.create_dataset("actions", data=actions)
    all_actions.append(actions)
    
    # 建立 obs group
    obs_grp = ep_grp.create_group("obs")
    obs_grp.create_dataset("lowdim_qpos", data=data['lowdim_qpos'])
    obs_grp.create_dataset("lowdim_ee", data=data['lowdim_ee'])
    obs_grp.create_dataset("language_instruction", data=data['language_instruction'].astype('S80'))
    
    # 儲存長度
    ep_grp.attrs["num_samples"] = len(actions)

# 計算 action 統計
all_actions = np.concatenate(all_actions, axis=0)
stats = {
    "action": {
        "min": all_actions.min(axis=0),
        "max": all_actions.max(axis=0),
        "mean": all_actions.mean(axis=0),
        "std": all_actions.std(axis=0)
    }
}

# 儲存統計到 pickle
stats_path = os.path.join(output_dir, "stats")
with open(stats_path, 'wb') as sf:
    pickle.dump(stats, sf)

# 關閉 HDF5
f.close()

print(f"\n✅ 轉換完成！")
print(f"   HDF5: {hdf5_path}")
print(f"   Stats: {stats_path}")
print(f"   Episodes: {len(pickle_files)}")
print(f"   Total actions: {len(all_actions)}")
