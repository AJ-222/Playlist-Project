import sys
import os
from myUtils.myGetters import print_track_info,get_all_h5_paths
# Dynamically add PythonSrc to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "PythonSrc"))
import hdf5_getters


root = "millionsongsubset/MillionSongSubset"  # Adjust this if your path differs
all_h5_files = get_all_h5_paths(root)

print(f"Found {len(all_h5_files)} HDF5 files.")
for path in all_h5_files[:5]:  # Preview first 5
    print(path)

song_data = []