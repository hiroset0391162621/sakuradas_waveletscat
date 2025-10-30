import datetime


#Fs_raw = 250.0
segment_duration_seconds = 60.0 #20.0
sampling_rate_hertz = 50.0
Fs = sampling_rate_hertz
overlap = 1 ### % overlap=1 means without overlapping
pooling = 'average' ### 'average' or 'max'


fiber = 'nojiri' #'round'

# hdf5_starttime_jst = datetime.datetime(2023, 12, 1, 0, 0, 0)
# hdf5_endttime_jst = datetime.datetime(2023, 12, 1, 0, 10, 0)

hdf5_starttime_jst = datetime.datetime(2025, 5, 30, 0, 0, 0)
hdf5_endttime_jst = datetime.datetime(2025, 5, 31, 0, 0, 0)

Nseconds = int( (hdf5_endttime_jst-hdf5_starttime_jst).total_seconds() )
N_minute = int( (hdf5_endttime_jst - hdf5_starttime_jst).total_seconds() / 60.0 )
windL = 60.0*N_minute
low_pass = 0.5
high_pass = 50

hdf5_dirname_base = f"/Volumes/Tohoku_15/Fiber-2_HDF5/"
hdf5_dirname = f"{hdf5_dirname_base}{hdf5_starttime_jst.year}/{hdf5_starttime_jst.month}/{hdf5_starttime_jst.day}/"  #"hdf5/"

used_channel_list = [str(_).zfill(4) for _ in range(700, 705, 5)]   

threshold = 90  ### dendrogram cut-off distance