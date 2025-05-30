import hdf5_getters
h5 = hdf5_getters.open_h5_file_read("millionsongsubset\MillionSongSubset\A\A\A\TRAAAAW128F429D538.h5")
duration = hdf5_getters.get_duration(h5)
h5.close()