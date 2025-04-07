import pickle
import glob
import sys
import numpy as np
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from obspy.core import UTCDateTime, Stream
import obspy
import datetime
#import spectral_func
from Params import *

sys.path.append("utils/")
from read_hdf5 import read_hdf5

sys.path.append("scat/")
from network import ScatteringNetwork

try:
    import scienceplots
except:
    pass
plt.style.use(["science", "nature"])
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.linewidth"] = 1.0  # 軸の太さを設定。目盛りは変わらない
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["xtick.minor.width"] = 0.8
plt.rcParams["ytick.minor.width"] = 0.8
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["xtick.major.size"] = 6.0
plt.rcParams["xtick.minor.size"] = 4.0
plt.rcParams["ytick.major.size"] = 6.0
plt.rcParams["ytick.minor.size"] = 4.0
plt.rcParams["xtick.major.pad"] = "8"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["axes.edgecolor"] = "#08192D"  # 枠の色
plt.rcParams["axes.labelcolor"] = "#08192D"  # labelの色
plt.rcParams["xtick.color"] = "#08192D"  # xticksの色
plt.rcParams["ytick.color"] = "#08192D"  # yticksの色
plt.rcParams["text.color"] = "#08192D"  # annotate, labelの色
plt.rcParams["legend.framealpha"] = 1.0  # legendの枠の透明度
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["text.usetex"] = False
plt.rcParams["date.converter"] = "concise"


if __name__ == "__main__":
    
    network_data = pickle.load(open("example/scattering_network.pickle", "rb"))
    
    
    N_minute = int( (hdf5_endttime_jst - hdf5_starttime_jst).total_seconds() / 60.0 )
    windL = 60.0*N_minute
    print("N_minute", N_minute)
    
    hdf5_starttime_utc = hdf5_starttime_jst + datetime.timedelta(hours=-9)
    hdf5_file_list = []
    for mm in range(N_minute):
        ts_utc = hdf5_starttime_utc + datetime.timedelta(minutes=mm)
        filename = glob.glob(
            hdf5_dirname+"decimator_"+ts_utc.strftime("%Y-%m-%d_%H.%M.%S")+"_UTC_"+"*.h5"
        )[0] 
        print(filename)     
        hdf5_file_list.append(filename)
        
    stream_minute = Stream()
    for i in range(len(hdf5_file_list)):
        stream_minute += read_hdf5(hdf5_file_list[i], fiber)
    
    stream_minute.merge(method=1)
    stream_minute.resample(Fs, no_filter=False, window="hann")
    
    print(stream_minute)

    stream_scat = Stream()
    print('channels', used_channel_list)
    for tr in stream_minute:
        if tr.stats.station in used_channel_list:
            stream_scat += tr.copy()
    
    stream_scat.plot(rasterized=True, equal_scale=False)