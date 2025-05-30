import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "PythonSrc"))
import hdf5_getters

def print_track_info(h5_path):
    h5 = hdf5_getters.open_h5_file_read(h5_path)

    # Basic Metadata
    title = hdf5_getters.get_title(h5).decode('utf-8')
    artist_name = hdf5_getters.get_artist_name(h5).decode('utf-8')
    artist_id = hdf5_getters.get_artist_id(h5).decode('utf-8')
    artist_mbid = hdf5_getters.get_artist_mbid(h5).decode('utf-8')
    release = hdf5_getters.get_release(h5).decode('utf-8')
    year = hdf5_getters.get_year(h5)
    duration = hdf5_getters.get_duration(h5)
    tempo = hdf5_getters.get_tempo(h5)
    key = hdf5_getters.get_key(h5)
    key_conf = hdf5_getters.get_key_confidence(h5)
    mode = hdf5_getters.get_mode(h5)
    mode_conf = hdf5_getters.get_mode_confidence(h5)
    time_sig = hdf5_getters.get_time_signature(h5)
    time_sig_conf = hdf5_getters.get_time_signature_confidence(h5)
    loudness = hdf5_getters.get_loudness(h5)
    artist_hotttnesss = hdf5_getters.get_artist_hotttnesss(h5)
    artist_familiarity = hdf5_getters.get_artist_familiarity(h5)
    song_hotttnesss = hdf5_getters.get_song_hotttnesss(h5)
    # Location Info
    location = hdf5_getters.get_artist_location(h5).decode('utf-8')
    latitude = hdf5_getters.get_artist_latitude(h5)
    longitude = hdf5_getters.get_artist_longitude(h5)

    h5.close()

    # Print info
    print("SONG METADATA")
    print(f"Title: {title}")
    print(f"Artist: {artist_name}")
    print(f"Release: {release}")
    print(f"Year: {year}")
    print(f"Duration: {duration:.2f} sec")
    print(f"Tempo: {tempo:.2f} BPM")
    print(f"Key: {key} (Confidence: {key_conf:.2f})")
    print(f"Mode: {'Major' if mode == 1 else 'Minor'} (Confidence: {mode_conf:.2f})")
    print(f"Time Signature: {time_sig} (Confidence: {time_sig_conf:.2f})")
    print(f"Loudness: {loudness:.2f} dB")
    print(f"Artist Familiarity: {artist_familiarity:.3f}")
    print(f"Artist Hotness: {artist_hotttnesss:.3f}")
    print(f"Song Hotness: {song_hotttnesss:.3f}")

    print("\nARTIST LOCATION")
    print(f"Location: {location}")
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")

def get_all_h5_paths(root_folder):
    h5_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".h5"):
                full_path = os.path.join(dirpath, file)
                h5_paths.append(full_path)
    return h5_paths
