import datetime


Fs_raw = 500.0
segment_duration_seconds = 10.0
sampling_rate_hertz = 100.0
Fs = sampling_rate_hertz
overlap = 1 ### % overlap=1 means without overlapping
pooling = 'max'


fiber = 'nojiri' #'round'

# hdf5_starttime_jst = datetime.datetime(2023, 12, 1, 0, 0, 0)
# hdf5_endttime_jst = datetime.datetime(2023, 12, 1, 0, 10, 0)

hdf5_starttime_jst = datetime.datetime(2024, 7, 14, 18, 15, 0)
hdf5_endttime_jst = datetime.datetime(2024, 7, 14, 18, 25, 0)

Nseconds = int( (hdf5_endttime_jst-hdf5_starttime_jst).total_seconds() )
N_minute = int( (hdf5_endttime_jst - hdf5_starttime_jst).total_seconds() / 60.0 )
windL = 60.0*N_minute
low_pass = 0.5
high_pass = 50

hdf5_dirname = "hdf5/"

used_channel_list = [str(_).zfill(4) for _ in range(500, 505, 5)]   
