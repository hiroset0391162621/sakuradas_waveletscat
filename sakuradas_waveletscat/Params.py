import datetime


Fs_raw = 250.0
segment_duration_seconds = 60.0
sampling_rate_hertz = 50.0
overlap = 1 ### % overlap=1 means without overlapping
scattering_coefficients_dirname = "J_1st5Q_1st4_J_2nd5Q_2nd2_60sec" ### Fs=50Hz, highpass0.5Hz, filterbank=[0.781Hz, 25Hz]
refstation = 'N.IJMV.U'
tlim = 60.0
pooling = 'max'


fiber = 'nojiri' #'round'

hdf5_starttime_jst = datetime.datetime(2023, 12, 1, 0, 0, 0)
hdf5_endttime_jst = datetime.datetime(2023, 12, 1, 0, 10, 0)

Fs = 50
low_pass = 0.2
high_pass = 50

hdf5_dirname = "hdf5/"

used_channel_list = [str(_).zfill(4) for _ in range(100, 105, 5)]   
