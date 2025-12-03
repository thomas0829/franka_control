#!/usr/bin/env python3
"""
Count video sessions in gello_teleop_video directory
Usage: python count_videos.py
"""
import os
from pathlib import Path
from collections import defaultdict


def count_video_sessions():
    """Count video sessions for each task in gello_teleop_video"""
    script_dir = Path(__file__).parent
    video_base_dir = script_dir / "gello_teleop_video"
    
    if not video_base_dir.exists():
        print(f"Video directory not found: {video_base_dir}")
        return
    
    # Count sessions for each task
    task_counts = defaultdict(int)
    total_sessions = 0
    
    for task_dir in sorted(video_base_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        
        task_name = task_dir.name
        session_count = 0
        
        # Count timestamped session directories (format: YYYYMMDD_HHMMSS)
        for item in task_dir.iterdir():
            if item.is_dir() and item.name.startswith("2025"):
                session_count += 1
        
        task_counts[task_name] = session_count
        total_sessions += session_count
    
    
    # Sort by session count (descending)
    sorted_tasks = sorted(task_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Task Name':<50} {'Sessions':>10}")
    print("-" * 70)
    
    for task_name, count in sorted_tasks:
        print(f"{task_name:<50} {count:>10}")
    
    print("=" * 70)
    
    # Group by session count
    print("\nGrouped by session count:")
    print("-" * 70)
    
    count_groups = defaultdict(list)
    for task_name, count in task_counts.items():
        count_groups[count].append(task_name)
    
    for count in sorted(count_groups.keys(), reverse=True):
        tasks = count_groups[count]
        print(f"\n{count} sessions ({len(tasks)} tasks):")
        for task in sorted(tasks):
            print(f"  - {task}")
    
    print()

    # Display results
    print("=" * 70)
    print(f"GELLO Video Sessions Summary")
    print("=" * 70)
    print(f"Total tasks: {len(task_counts)}")
    print(f"Total sessions: {total_sessions}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    count_video_sessions()
