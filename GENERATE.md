# How to generate multiple motion qpos files

1. Convert .bvh file to .csv (Where GMR comes in)
2. pkl_to_csv_converter.py to run 
```
usage: pkl_to_csv_converter.py [-h] [--pkl-file PKL_FILE | --input-dir INPUT_DIR] [--output-dir OUTPUT_DIR] [--csv-out CSV_OUT] [--fps FPS]
                               [--start-frame START_FRAME] [--end-frame END_FRAME] [--write-metadata]

Convert GMR .pkl motion files (qpos/fps) to CSV (qpos+qvel).

options:
  -h, --help            show this help message and exit
  --pkl-file PKL_FILE   Convert a single .pkl file
  --input-dir INPUT_DIR
                        Directory containing .pkl files (batch mode)
  --output-dir OUTPUT_DIR
                        Directory to write .csv files (batch mode, or single-file default)
  --csv-out CSV_OUT     Output CSV path (single-file mode only)
  --fps FPS             Target FPS for output CSV (default: 50)
  --start-frame START_FRAME
                        Start frame index (inclusive) in the original sequence
  --end-frame END_FRAME
                        End frame index (inclusive) in the original sequence
  --write-metadata      Write frame-range metadata as commented header lines in CSV

Examples:
  # Batch convert all .pkl under a directory
  python scripts/pkl_to_csv_converter.py --input-dir gmr_output/lafan1 --output-dir gmr_output/lafan1/csv

  # Convert a single file
  python scripts/pkl_to_csv_converter.py --pkl-file gmr_output/lafan1/foo.pkl --output-dir gmr_output/lafan1/csv

  # Resample to a different FPS
  python scripts/pkl_to_csv_converter.py --pkl-file foo.pkl --fps 100

  # Slice frames (inclusive, in original indices)
  python scripts/pkl_to_csv_converter.py --pkl-file foo.pkl --start-frame 100 --end-frame 400

Notes:
  - Output CSV has 71 columns: qpos(36) + qvel(35).
  - If --start-frame/--end-frame is provided, metadata header is written by default
    (equivalent to passing --write-metadata).
```