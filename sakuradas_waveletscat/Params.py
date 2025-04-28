import datetime


#Fs_raw = 250.0
segment_duration_seconds = 20.0
sampling_rate_hertz = 100.0
Fs = sampling_rate_hertz
overlap = 1 ### % overlap=1 means without overlapping
pooling = 'max'


fiber = 'nojiri' #'round'

# hdf5_starttime_jst = datetime.datetime(2023, 12, 1, 0, 0, 0)
# hdf5_endttime_jst = datetime.datetime(2023, 12, 1, 0, 10, 0)

hdf5_starttime_jst = datetime.datetime(2025, 3, 27, 15, 0, 0)
hdf5_endttime_jst = datetime.datetime(2025, 3, 27, 16, 0, 0)

Nseconds = int( (hdf5_endttime_jst-hdf5_starttime_jst).total_seconds() )
N_minute = int( (hdf5_endttime_jst - hdf5_starttime_jst).total_seconds() / 60.0 )
windL = 60.0*N_minute
low_pass = 0.5
high_pass = 50

hdf5_dirname = "/Users/hirosetakashi/Volumes/noise_monitoring/noise_monitoring/DAS/Tohoku_15/Fiber-2_HDF5/2025/03/27/"  #"hdf5/"

used_channel_list = [str(_).zfill(4) for _ in range(805, 890, 5)]   
