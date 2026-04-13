#!/usr/bin/env python3

"""pkl_to_csv_converter.py

Converts GMR motion .pkl files (containing `qpos` and optional `fps`) into a
CSV layout compatible with downstream replay scripts:

- Per-row layout: `qpos(36) + qvel(35) = 71 columns`
    - qpos: root_pos(3) + root_quat_wxyz(4) + joint_q(29)
    - qvel: root_vel(3) + root_ang_vel(3) + joint_qd(29)

Quick usage:

Batch convert a directory of .pkl files (default paths):
    python scripts/pkl_to_csv_converter.py --input-dir gmr_output/lafan1 --output-dir gmr_output/lafan1/csv

Convert a single file:
    python scripts/pkl_to_csv_converter.py --pkl-file gmr_output/lafan1/foo.pkl --output-dir gmr_output/lafan1/csv

Resample to a different FPS:
    python scripts/pkl_to_csv_converter.py --pkl-file foo.pkl --fps 100

Slice a frame range (inclusive, in ORIGINAL frames, before resampling):
    python scripts/pkl_to_csv_converter.py --pkl-file foo.pkl --start-frame 100 --end-frame 400

Notes:
- If you pass --start-frame/--end-frame, the script writes a commented metadata
    header by default (same as passing --write-metadata).
"""

import sys
import pickle
import numpy as np
from pathlib import Path
import argparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "gmr_output" / "lafan1"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "csv"


def _load_motion_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

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

def _clamp_frame_range(num_frames, start_frame=None, end_frame=None):
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = num_frames - 1

    start_frame = int(start_frame)
    end_frame = int(end_frame)

    if start_frame < 0:
        start_frame = 0
    if end_frame < 0:
        end_frame = num_frames - 1

    if start_frame >= num_frames:
        raise ValueError(f"start_frame ({start_frame}) must be < num_frames ({num_frames})")
    if end_frame >= num_frames:
        end_frame = num_frames - 1
    if end_frame < start_frame:
        raise ValueError(f"end_frame ({end_frame}) must be >= start_frame ({start_frame})")

    return start_frame, end_frame


def pkl_to_csv(pkl_path, csv_path, target_fps=50, start_frame=None, end_frame=None, write_metadata=False, data=None):
    """
    Convert pkl motion data to CSV format
    
    Args:
        pkl_path: Path to input pkl file
        csv_path: Path to output CSV file
        target_fps: Target frame rate for output (default 50Hz)
        start_frame: Optional start frame index (inclusive) in the original sequence
        end_frame: Optional end frame index (inclusive) in the original sequence
        write_metadata: If True, writes frame-range metadata as commented header lines
    """
    # Load pkl data
    if data is None:
        data = _load_motion_pkl(pkl_path)
    
    qpos_list = data['qpos']
    original_fps = data.get('fps', 30)
    
    print(f"  📊 Loaded {len(qpos_list)} frames at {original_fps} FPS")

    # Frame slicing (in original frame indices, before resampling)
    original_num_frames = len(qpos_list)
    if start_frame is not None or end_frame is not None:
        start_frame_i, end_frame_i = _clamp_frame_range(original_num_frames, start_frame, end_frame)
        qpos_list = qpos_list[start_frame_i:end_frame_i + 1]
        print(
            f"  ✂️  Using frames {start_frame_i}..{end_frame_i} (inclusive) "
            f"-> {len(qpos_list)} frames"
        )
    else:
        start_frame_i, end_frame_i = 0, original_num_frames - 1

    if len(qpos_list) < 2:
        raise ValueError(f"Need at least 2 frames after slicing (got {len(qpos_list)})")
    
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
    
    # Save, optionally with commented metadata header lines.
    header = ""
    if write_metadata:
        header_lines = [
            f"source_pkl={Path(pkl_path).name}",
            f"original_fps={original_fps}",
            f"target_fps={target_fps}",
            f"original_total_frames={original_num_frames}",
            f"selected_start_frame={start_frame_i}",
            f"selected_end_frame={end_frame_i}",
            f"selected_frames={len(qpos_list)}",
            f"columns={csv_array.shape[1]}",
        ]
        header = "\n".join(header_lines)

    # Proper precision for float32
    np.savetxt(csv_path, csv_array, delimiter=',', fmt='%.6f', header=header, comments='# ')
    
    print(f"  ✅ Saved {len(csv_data)} frames to {csv_path}")
    return len(csv_data)

def batch_convert_pkl_to_csv(input_dir, output_dir, target_fps=50, start_frame=None, end_frame=None, write_metadata=False):
    """Convert all pkl files in a directory to CSV format"""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
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
        
        try:
            data = _load_motion_pkl(pkl_path)
            qpos_list = data['qpos']
            original_num_frames = len(qpos_list)

            # Generate CSV filename (include slicing info if provided)
            if start_frame is not None or end_frame is not None:
                start_frame_i, end_frame_i = _clamp_frame_range(original_num_frames, start_frame, end_frame)
                csv_name = f"{pkl_path.stem}_start_{start_frame_i}_end_{end_frame_i}.csv"
            else:
                csv_name = pkl_path.name.replace('.pkl', '.csv')

            csv_path = output_dir / csv_name

            frame_count = pkl_to_csv(
                pkl_path,
                csv_path,
                target_fps=target_fps,
                start_frame=start_frame,
                end_frame=end_frame,
                write_metadata=write_metadata,
                data=data,
            )
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


def convert_single_pkl_to_csv(pkl_file, csv_out=None, output_dir=None, target_fps=50, start_frame=None, end_frame=None, write_metadata=False):
    """Convert a single pkl file to CSV format."""

    pkl_path = Path(pkl_file)
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")

    data = _load_motion_pkl(pkl_path)
    original_num_frames = len(data['qpos'])

    if csv_out is not None:
        csv_path = Path(csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(output_dir) if output_dir is not None else pkl_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if start_frame is not None or end_frame is not None:
            start_frame_i, end_frame_i = _clamp_frame_range(original_num_frames, start_frame, end_frame)
            csv_name = f"{pkl_path.stem}_start_{start_frame_i}_end_{end_frame_i}.csv"
        else:
            csv_name = pkl_path.name.replace('.pkl', '.csv')

        csv_path = out_dir / csv_name

    print(f"\n📁 Processing single file: {pkl_path.name}")
    pkl_to_csv(
        pkl_path,
        csv_path,
        target_fps=target_fps,
        start_frame=start_frame,
        end_frame=end_frame,
        write_metadata=write_metadata,
        data=data,
    )
    return csv_path


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Convert GMR .pkl motion files (qpos/fps) to CSV (qpos+qvel).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Batch convert all .pkl under a directory\n"
            "  python scripts/pkl_to_csv_converter.py --input-dir gmr_output/lafan1 --output-dir gmr_output/lafan1/csv\n\n"
            "  # Convert a single file\n"
            "  python scripts/pkl_to_csv_converter.py --pkl-file gmr_output/lafan1/foo.pkl --output-dir gmr_output/lafan1/csv\n\n"
            "  # Resample to a different FPS\n"
            "  python scripts/pkl_to_csv_converter.py --pkl-file foo.pkl --fps 100\n\n"
            "  # Slice frames (inclusive, in original indices)\n"
            "  python scripts/pkl_to_csv_converter.py --pkl-file foo.pkl --start-frame 100 --end-frame 400\n\n"
            "Notes:\n"
            "  - Output CSV has 71 columns: qpos(36) + qvel(35).\n"
            "  - If --start-frame/--end-frame is provided, metadata header is written by default\n"
            "    (equivalent to passing --write-metadata).\n"
        ),
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--pkl-file",
        type=str,
        default=None,
        help="Convert a single .pkl file",
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing .pkl files (batch mode)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write .csv files (batch mode, or single-file default)",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Output CSV path (single-file mode only)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50,
        help="Target FPS for output CSV (default: 50)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Start frame index (inclusive) in the original sequence",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End frame index (inclusive) in the original sequence",
    )
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Write frame-range metadata as commented header lines in CSV",
    )

    return parser.parse_args(argv)

if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    # If user is slicing, default to writing metadata unless explicitly disabled.
    write_metadata = args.write_metadata or (args.start_frame is not None or args.end_frame is not None)

    if args.pkl_file is not None:
        if args.csv_out is not None and args.output_dir is not None:
            # output_dir is ignored when csv_out is explicitly provided.
            pass
        convert_single_pkl_to_csv(
            pkl_file=args.pkl_file,
            csv_out=args.csv_out,
            output_dir=args.output_dir,
            target_fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            write_metadata=write_metadata,
        )
    else:
        batch_convert_pkl_to_csv(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            write_metadata=write_metadata,
        )