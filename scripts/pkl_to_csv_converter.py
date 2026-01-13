#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
from pathlib import Path

def quat_to_angular_velocity(quat_prev, quat_curr, dt):
    """
    Convert quaternion difference to angular velocity
    
    Args:
        quat_prev: Previous quaternion [qw, qx, qy, qz] (WXYZ format)
        quat_curr: Current quaternion [qw, qx, qy, qz] (WXYZ format)  
        dt: Time step
        
    Returns:
        Angular velocity [wx, wy, wz]
    """
    # Normalize quaternions
    quat_prev = quat_prev / np.linalg.norm(quat_prev)
    quat_curr = quat_curr / np.linalg.norm(quat_curr)
    
    # Quaternions are already in WXYZ format - use directly
    q1 = quat_prev  # [w,x,y,z]
    q2 = quat_curr  # [w,x,y,z]
    
    # Compute relative rotation: q_rel = q2 * q1^-1
    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quat_inverse(q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z])
    
    q_rel = quat_multiply(q2, quat_inverse(q1))
    
    # Convert quaternion to angular velocity
    # For small rotations: ω ≈ 2 * [x, y, z] / dt (when w ≈ 1)
    if q_rel[0] < 0:  # Ensure shortest path
        q_rel = -q_rel
        
    ang_vel = 2.0 * q_rel[1:4] / dt
    return ang_vel

def compute_velocities(qpos_data, dt):
    """
    Compute velocities from position data using central difference
    Matches the format expected by replay_g1_standup.py
    
    Args:
        qpos_data: List of position arrays, each with 36 values
        dt: Time step between frames
    
    Returns:
        List of velocity arrays, each with 35 values
    """
    qvel_data = []
    
    for i in range(len(qpos_data)):
        # Get current frame data
        qpos_curr = np.array(qpos_data[i])
        
        # Compute linear velocities using finite differences
        if i == 0:
            # Forward difference for first frame
            qpos_next = np.array(qpos_data[i + 1])
            root_pos_vel = (qpos_next[:3] - qpos_curr[:3]) / dt
            joint_vel = (qpos_next[7:] - qpos_curr[7:]) / dt
            
            # Angular velocity from quaternion difference
            quat_curr = qpos_curr[3:7]  # [qw, qx, qy, qz] (WXYZ format)
            quat_next = qpos_next[3:7]
            root_ang_vel = quat_to_angular_velocity(quat_curr, quat_next, dt)
            
        elif i == len(qpos_data) - 1:
            # Backward difference for last frame
            qpos_prev = np.array(qpos_data[i - 1])
            root_pos_vel = (qpos_curr[:3] - qpos_prev[:3]) / dt
            joint_vel = (qpos_curr[7:] - qpos_prev[7:]) / dt
            
            # Angular velocity from quaternion difference
            quat_prev = qpos_prev[3:7]  # [qw, qx, qy, qz] (WXYZ format)
            quat_curr = qpos_curr[3:7]
            root_ang_vel = quat_to_angular_velocity(quat_prev, quat_curr, dt)
            
        else:
            # Central difference for middle frames
            qpos_prev = np.array(qpos_data[i - 1])
            qpos_next = np.array(qpos_data[i + 1])
            root_pos_vel = (qpos_next[:3] - qpos_prev[:3]) / (2 * dt)
            joint_vel = (qpos_next[7:] - qpos_prev[7:]) / (2 * dt)
            
            # Angular velocity from quaternion difference
            quat_prev = qpos_prev[3:7]  # [qw, qx, qy, qz] (WXYZ format)
            quat_next = qpos_next[3:7]
            root_ang_vel = quat_to_angular_velocity(quat_prev, quat_next, 2 * dt)
        
        # Ensure we have exactly 35 velocity components
        # qvel format: root_vel(3) + root_ang_vel(3) + joint_qd(29) = 35 total
        assert len(root_pos_vel) == 3, f"Expected 3 root position velocities, got {len(root_pos_vel)}"
        assert len(root_ang_vel) == 3, f"Expected 3 root angular velocities, got {len(root_ang_vel)}"
        assert len(joint_vel) == 29, f"Expected 29 joint velocities, got {len(joint_vel)}"
        
        # Combine: 3 + 3 + 29 = 35 total
        full_qvel = np.concatenate([root_pos_vel, root_ang_vel, joint_vel])
        qvel_data.append(full_qvel.astype(np.float32))
    
    return qvel_data

def pkl_to_csv(pkl_path, csv_path, target_fps=50):
    """
    Convert pkl motion data to CSV format
    
    Args:
        pkl_path: Path to input pkl file
        csv_path: Path to output CSV file
        target_fps: Target frame rate for output (default 50Hz)
    """
    # Load pkl data
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    qpos_list = data['qpos']
    original_fps = data.get('fps', 50)
    
    print(f"  📊 Loaded {len(qpos_list)} frames at {original_fps} FPS")
    
    # Resample if needed
    if original_fps != target_fps:
        print(f"  🔄 Resampling from {original_fps} Hz to {target_fps} Hz")
        
        # Create time arrays
        original_times = np.linspace(0, len(qpos_list) / original_fps, len(qpos_list))
        target_length = int(len(qpos_list) * target_fps / original_fps)
        target_times = np.linspace(0, len(qpos_list) / original_fps, target_length)
        
        # Interpolate each component
        qpos_array = np.array(qpos_list)
        resampled_qpos = []
        
        for i in range(qpos_array.shape[1]):  # For each DOF
            interp_values = np.interp(target_times, original_times, qpos_array[:, i])
            resampled_qpos.append(interp_values)
        
        qpos_array = np.array(resampled_qpos).T
        qpos_list = [qpos_array[i] for i in range(len(qpos_array))]
    
    # Compute velocities
    dt = 1.0 / target_fps
    print(f"  🧮 Computing velocities with dt={dt:.6f}s")
    qvel_list = compute_velocities(qpos_list, dt)
    
    # Combine qpos and qvel into final format
    print(f"  🔗 Combining position and velocity data")
    csv_data = []
    
    for i in range(len(qpos_list)):
        qpos = np.array(qpos_list[i], dtype=np.float32)  # 36 values
        qvel = np.array(qvel_list[i], dtype=np.float32)  # 35 values
        
        # Ensure we have exactly the right dimensions
        assert len(qpos) == 36, f"qpos should have 36 values, got {len(qpos)}"
        assert len(qvel) == 35, f"qvel should have 35 values, got {len(qvel)}"
        
        # Verify qpos format: root_pos(3) + root_quat_wxyz(4) + joint_q(29)
        root_pos = qpos[:3]
        root_quat = qpos[3:7]  # Should be [qw, qx, qy, qz] format (WXYZ)
        joint_q = qpos[7:36]   # 29 joint positions
        
        # Verify qvel format: root_vel(3) + root_ang_vel(3) + joint_qd(29)  
        root_vel = qvel[:3]
        root_ang_vel = qvel[3:6]
        joint_qd = qvel[6:35]   # 29 joint velocities
        
        # Combine: 36 + 35 = 71 total columns (matches replay_g1_standup.py expectation)
        row = np.concatenate([qpos, qvel])
        csv_data.append(row)
    
    # Convert to numpy array and save as CSV
    csv_array = np.array(csv_data, dtype=np.float32)
    print(f"  💾 Saving CSV with shape {csv_array.shape}")
    
    # Verify final format matches replay_g1_standup.py expectations
    expected_width = 36 + 35  # ROOT_QPOS_WIDTH + joint_dofs + (3 + 3 + joint_dofs)
    if csv_array.shape[1] != expected_width:
        raise ValueError(f"CSV must have {expected_width} columns (got {csv_array.shape[1]})")
    
    # Save without headers, proper precision for float32
    np.savetxt(csv_path, csv_array, delimiter=',', fmt='%.6f')
    
    print(f"  ✅ Saved {len(csv_data)} frames to {csv_path}")
    return len(csv_data)

def batch_convert_pkl_to_csv():
    """Convert all pkl files in gmr_output to CSV format"""
    
    input_dir = Path("/home/jaeryeong/GMR/gmr_output/lafan1")
    output_dir = Path("/home/jaeryeong/GMR/gmr_output/lafan1/csv")
    output_dir.mkdir(exist_ok=True)
    
    pkl_files = list(input_dir.glob("*.pkl"))
    
    if not pkl_files:
        print("❌ No pkl files found in input directory")
        return
    
    print(f"🔄 Converting {len(pkl_files)} pkl files to CSV format")
    print("="*80)
    
    total_frames = 0
    converted_files = 0
    
    for pkl_path in sorted(pkl_files):
        print(f"\n📁 Processing: {pkl_path.name}")
        
        # Generate CSV filename
        csv_name = pkl_path.name.replace('.pkl', '.csv')
        csv_path = output_dir / csv_name
        
        try:
            frame_count = pkl_to_csv(pkl_path, csv_path, target_fps=50)
            total_frames += frame_count
            converted_files += 1
            
        except Exception as e:
            print(f"  ❌ Error converting {pkl_path.name}: {e}")
            continue
    
    print(f"\n🎉 Conversion completed!")
    print(f"  📊 Converted {converted_files}/{len(pkl_files)} files")
    print(f"  🎬 Total frames processed: {total_frames:,}")
    print(f"  📂 CSV files saved to: {output_dir}")
    
    # List generated CSV files
    csv_files = list(output_dir.glob("*.csv"))
    print(f"\n📋 Generated CSV files:")
    for csv_file in sorted(csv_files):
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"  💾 {csv_file.name} ({file_size_mb:.1f} MB)")

if __name__ == "__main__":
    batch_convert_pkl_to_csv()