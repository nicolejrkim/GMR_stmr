#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, '/home/jaeryeong/GMR')

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

def batch_retarget_lafan1():
    """Process multiple LAFAN1 motions to different robots"""
    
    lafan1_dir = Path("/home/jaeryeong/GMR/motion_data/lafan1")
    output_dir = Path("/home/jaeryeong/GMR/gmr_output/lafan1")
    output_dir.mkdir(exist_ok=True)
    
    # Select some interesting motions
    test_motions = [
        # "dance1_subject1.bvh",
        # "dance2_subject1.bvh", 
        # "walk1_subject1.bvh",
        # "run1_subject2.bvh",
        # "jumps1_subject1.bvh",
        # "fight1_subject2.bvh"
        "fallAndGetUp2_subject2.bvh"
    ]
    
    # Test robots that support LAFAN1 (from README table)
    test_robots = [
        "unitree_g1"
    ]
    
    print(f"🎯 Processing {len(test_motions)} motions × {len(test_robots)} robots = {len(test_motions) * len(test_robots)} combinations")
    print("="*80)
    
    for motion_file in test_motions:
        bvh_path = lafan1_dir / motion_file
        if not bvh_path.exists():
            print(f"❌ Skipping {motion_file} - file not found")
            continue
            
        print(f"\n🎭 Processing motion: {motion_file}")
        
        # Load motion data once
        try:
            lafan1_data_frames, actual_human_height = load_bvh_file(str(bvh_path), format="lafan1")
            print(f"   📊 Loaded {len(lafan1_data_frames)} frames, human height: {actual_human_height:.2f}m")
        except Exception as e:
            print(f"❌ Error loading {motion_file}: {e}")
            continue
        
        for robot_name in test_robots:
            try:
                print(f"   🤖 Retargeting to {robot_name}...")
                
                # Initialize retargeter
                retargeter = GMR(
                    src_human="bvh_lafan1",
                    tgt_robot=robot_name,
                    actual_human_height=actual_human_height,
                )
                
                # Process all frames
                qpos_list = []
                for i in tqdm(range(len(lafan1_data_frames)), desc="   Processing frames", leave=False):
                    frame_data = lafan1_data_frames[i]
                    robot_qpos = retargeter.retarget(frame_data)
                    qpos_list.append(robot_qpos.copy())
                
                # Save results
                motion_name = motion_file.replace('.bvh', '')
                save_path = output_dir / f"{motion_name}_{robot_name}_gmr.pkl"
                
                motion_data = {
                    'qpos': qpos_list,
                    'fps': 50,
                    'robot_type': robot_name,
                    'source_motion': str(bvh_path),
                    'human_height': actual_human_height,
                    'num_frames': len(qpos_list)
                }
                
                with open(save_path, 'wb') as f:
                    pickle.dump(motion_data, f)
                
                print(f"   ✅ Saved {len(qpos_list)} frames to {save_path.name}")
                
            except Exception as e:
                print(f"   ❌ Error with {robot_name}: {e}")
                continue
    
    # List all generated files
    generated_files = list(output_dir.glob("*.pkl"))
    print(f"\n🎉 Successfully generated {len(generated_files)} motion files:")
    for f in sorted(generated_files):
        print(f"   📁 {f.name}")
    
    print("\n✨ Batch processing completed!")
    print(f"📂 All files saved to: {output_dir}")

if __name__ == "__main__":
    batch_retarget_lafan1()