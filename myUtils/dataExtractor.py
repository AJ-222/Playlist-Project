import csv
import os
import sys
import numpy as np
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "PythonSrc"))
import hdf5_getters

def extract_metadata_with_vectors(h5_path):
    try:
        h5 = hdf5_getters.open_h5_file_read(h5_path)

        title = hdf5_getters.get_title(h5).decode("utf-8")
        artist = hdf5_getters.get_artist_name(h5).decode("utf-8")

        scalar_features = [
            hdf5_getters.get_tempo(h5),
            hdf5_getters.get_loudness(h5),
            hdf5_getters.get_energy(h5),
            hdf5_getters.get_danceability(h5),
            hdf5_getters.get_duration(h5),
            hdf5_getters.get_key(h5),
            hdf5_getters.get_mode(h5),
            hdf5_getters.get_year(h5),
            hdf5_getters.get_song_hotttnesss(h5)
        ]

        timbre = hdf5_getters.get_segments_timbre(h5)
        pitches = hdf5_getters.get_segments_pitches(h5)
        segment_starts = hdf5_getters.get_segments_start(h5)
        section_starts = hdf5_getters.get_sections_start(h5)

        # Average over sections → single scalar per feature group
        section_timbres = []
        section_pitches = []

        for i in range(len(section_starts)):
            start = section_starts[i]
            end = section_starts[i + 1] if i + 1 < len(section_starts) else float("inf")

            indices = [j for j, t in enumerate(segment_starts) if start <= t < end]
            if indices:
                section_timbres.append(np.mean(timbre[indices]))
                section_pitches.append(np.mean(pitches[indices]))

        avg_timbre = np.mean(section_timbres) if section_timbres else 0.0
        avg_pitches = np.mean(section_pitches) if section_pitches else 0.0

        # Filter artist terms
        terms = hdf5_getters.get_artist_terms(h5)
        term_weights = hdf5_getters.get_artist_terms_weight(h5)
        term_freqs = hdf5_getters.get_artist_terms_freq(h5)

        filtered_terms = [
            f"{terms[i].decode('utf-8')}:{term_weights[i]:.3f}"
            for i in range(len(terms))
            if term_freqs[i] > 0.2
        ]

        h5.close()

        return {
            "Title": title,
            "Artist": artist,
            "Features": scalar_features + [avg_timbre, avg_pitches],
            "FilteredTerms": ";".join(filtered_terms)
        }

    except Exception as e:
        print(f"❌ Error processing {h5_path}: {e}")
        return None


def process_all_to_csv(input_csv="h5_paths.csv", output_csv="track_raw_features.csv"):
    with open(input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        paths = [row["H5Path"] for row in reader]
    scalar_names = ["Tempo", "Loudness", "Energy", "Danceability", "Duration", "Key", "Mode", "Year", "Hotttnesss"]
    extra_names = ["AvgTimbre", "AvgPitches"]
    fieldnames = ["Title", "Artist"] + scalar_names + extra_names + ["FilteredTerms"]

    with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        for path in paths:
            result = extract_metadata_with_vectors(path)
            if result:
                row = {
                    "Title": result["Title"],
                    "Artist": result["Artist"],
                    "FilteredTerms": result["FilteredTerms"]
                }
                row.update({
                    fieldnames[i + 2]: result["Features"][i]
                    for i in range(len(result["Features"]))
                })
                writer.writerow(row)
                count += 1

    print(f"✅ Processed {count} songs with averaged section features and filtered terms → {output_csv}")


# === Run it ===
process_all_to_csv("h5_paths.csv", "track_raw_features.csv")
