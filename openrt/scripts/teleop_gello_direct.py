#!/usr/bin/env python3
"""
GELLO → Franka 直接控制（繞過環境）
使用 update_joints 直接發送關節指令
"""
import sys
import time
import numpy as np

sys.path.insert(0, '/home/robots/yuquand/polymetis_franka')

from robot.controllers.gello_controller import GelloController
from robot.real.server_interface import ServerInterface


def main():
    print("=" * 70)
    print(" " * 15 + "GELLO → Franka 直接控制")
    print("=" * 70)
    
    nuc_ip = "192.168.1.6"
    
    # 1. Initialize GELLO
    print("\n[1/2] GELLO 初始化...")
    gello = GelloController(port='/dev/ttyUSB0', baudrate=57600)
    print("✅ GELLO ready")
    
    # 2. Connect to Franka via NUC
    print(f"\n[2/2] 連接 Franka ({nuc_ip})...")
    robot = ServerInterface(ip_address=nuc_ip, launch=True)
    
    # 讀取初始狀態（保存用於恢復）
    initial_robot_joints = robot.get_joint_positions()
    robot_joints = initial_robot_joints.copy()
    print(f"   Franka 當前位置: {robot_joints.round(3)}")
    
    print("\n" + "="*70)
    print("⚠️  請將 GELLO 擺成和 Franka 一樣的姿態")
    print("   (看著 Franka，把 GELLO 擺成相同的手臂形狀)")
    print("   擺好後按 Enter 鍵繼續...")
    print("="*70)
    input()
    
    # 自動校準：讀取當前 GELLO 原始數據，計算 offset
    print("\n正在自動校準 GELLO offset...")
    gello_raw_rad = gello.driver.get_joints()  # 讀取數據（已經轉換為 rad）
    gello_raw_rad = np.array(gello_raw_rad[:7])
    
    # 計算新的 offset：考慮 joint_signs
    # 公式：(raw - offset) * sign = target
    # 所以：offset = raw - (target / sign)
    new_offset = gello_raw_rad - (robot_joints / gello.joint_signs)
    print(f"   GELLO 原始讀數: {gello_raw_rad.round(3)}")
    print(f"   Joint signs: {gello.joint_signs}")
    print(f"   計算出的 offset: {new_offset.round(3)}")
    
    # 更新 GELLO controller 的 offset
    gello.joint_offsets = new_offset
    
    # 驗證校準
    gello_calibrated = gello.get_joint_state()[:7]
    diff = np.abs(gello_calibrated - robot_joints).max()
    print(f"   校準後誤差: {diff:.3f} rad ({diff*57.3:.1f}°)")
    
    if diff > 0.2:
        print("⚠️  警告：校準誤差較大，請確認 GELLO 和 Franka 姿態是否一致")
        print("   按 Enter 繼續，或 Ctrl+C 取消...")
        input()
    else:
        print("✅ 校準成功！\n")
    
    # 先發送一個命令啟動 controller
    print("啟動 joint controller...")
    robot.update_joints(robot_joints.tolist(), velocity=False, blocking=True)
    time.sleep(1.0)
    
    print("="*70)
    print("✅ 開始 Teleoperation!")
    print("   移動 GELLO 來控制 Franka")
    print("   按 Ctrl+C 停止")
    print("="*70 + "\n")
    
    # 3. Control loop
    loop_count = 0
    
    try:
        while True:
            # Read GELLO
            gello_state = gello.get_joint_state()
            target_joints = gello_state[:7]
            gello_gripper = gello_state[7]  # 0=open, 1=closed
            
            # Send joints to robot (joint position control)
            robot.update_joints(target_joints.tolist(), velocity=False, blocking=False)
            
            # Send gripper command
            # gello_gripper: 0=open, 1=closed
            # NUC will handle: width = max_width * (1 - command)
            robot.update_gripper(gello_gripper, velocity=False, blocking=False)
            
            # Display
            if loop_count % 20 == 0:
                robot_joints = robot.get_joint_positions()
                diff = np.linalg.norm(target_joints - robot_joints)
                print(f"[{loop_count:04d}] Diff: {diff:.3f} | "
                      f"GELLO J1: {target_joints[0]:.2f} | "
                      f"Robot J1: {robot_joints[0]:.2f} | "
                      f"Gripper: {gello_gripper:.2f}")
            
            loop_count += 1
            time.sleep(0.05)  # 20Hz
            
    except KeyboardInterrupt:
        print("\n\n⚠️  停止中...")
    finally:
        print("\n正在恢復到初始位置...")
        try:
            # 先打開夾爪
            robot.update_gripper(0.0, velocity=False, blocking=True)
            time.sleep(0.5)
            # 恢復關節位置
            robot.update_joints(initial_robot_joints.tolist(), velocity=False, blocking=True)
            print("✅ 已恢復到初始位置")
        except Exception as e:
            print(f"⚠️  恢復時發生錯誤: {e}")
        
        print("\nCleanup...")
        gello.close()
        print("✅ Done")


if __name__ == "__main__":
    main()
