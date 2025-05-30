import os
import csv

def get_all_h5_paths(root_folder):
    h5_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".h5"):
                full_path = os.path.join(dirpath, file)
                h5_paths.append(full_path)
    return h5_paths

def save_paths_to_csv(h5_paths, output_file="h5_paths.csv"):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["H5Path"])  # CSV header
        for path in h5_paths:
            writer.writerow([path])
    print(f"✅ Saved {len(h5_paths)} paths to {output_file}")

# === Run it ===
root = "millionsongsubset/MillionSongSubset"  # Adjust path if needed
all_paths = get_all_h5_paths(root)
save_paths_to_csv(all_paths)