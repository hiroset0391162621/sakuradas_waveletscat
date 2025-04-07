import datetime
starttime = datetime.datetime(2023,10,10,0)
endtime = datetime.datetime(2023,11,30,23)

Fs_raw = 250.0
segment_duration_seconds = 60.0
sampling_rate_hertz = 50.0
overlap = 1 ### % overlap=1 means without overlapping
scattering_coefficients_dirname = "J_1st5Q_1st4_J_2nd5Q_2nd2_60sec" ### Fs=50Hz, highpass0.5Hz, filterbank=[0.781Hz, 25Hz]
refstation = 'N.IJMV.U'
tlim = 60.0
pooling = 'max'
comps = ['U', 'E', 'N']
