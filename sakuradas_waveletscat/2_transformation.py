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
import spectral_func

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


def cosTaper(windL, percent):
    N = windL
    tp = np.ones(N)
    for i in range(int(N*percent+1)):
        tp[i] *= 0.5 * (1 - np.cos((np.pi * i) / ( N * percent)))

    for i in range(int(N*(1-percent)), N):
        tp[i] *= 0.5 * (1 - np.cos((np.pi * (i+1)) / ( N * percent)))

    return tp

if __name__ == "__main__":
    
    network_data = pickle.load(open("example/scattering_network.pickle", "rb"))
    
    
    
    
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
    
    #stream_scat.plot(rasterized=True, equal_scale=False)
    
    
    # Extract segment length (from any layer)
    segment_duration = network_data.bins / network_data.sampling_rate
    

    #overlap = 1 ### % overlap=1 means without overlapping
    print('segment_duration', segment_duration)
    Nch = len(stream_scat)

    tp = cosTaper(network_data.bins, 0.05)

    # Gather list for timestamps and segments
    timestamps = list()
    segments = list()
    # Collect data and timestamps
    for traces in stream_scat.slide(segment_duration, segment_duration * overlap):
        timestamps.append(mdates.num2date(traces[0].times(type="matplotlib")[0])+datetime.timedelta(seconds=segment_duration_seconds*0.5))
        
        traces_sub = np.array([trace.data[:-1] for trace in traces])
        
        
        
        if traces_sub.shape[1]!= network_data.bins:
            padd = network_data.bins - traces_sub.shape[1]
            print(Nch, padd)
            traces_sub = np.concatenate((traces_sub, np.zeros((Nch,padd))), axis=1)
            
        # if np.nanmax(np.abs(traces_sub))<100:
        #     #print('all zero', mdates.num2date(traces[0].times(type="matplotlib")[0]))
        #     traces_sub *= np.nan
        
        traces_sub *= tp
        
        
        
        segments.append(traces_sub)
    
    
    scattering_coefficients = network_data.transform(segments, reduce_type=np.max)
    
    # Extract the first channel
    channel_id = 0
    trace = stream_scat[channel_id]
    

    order_1 = np.log10(scattering_coefficients[0][:, channel_id, :].squeeze())
    center_frequencies = network_data.banks[0].centers


    # Create figure and axes
    fig, ax = plt.subplots(2, sharex=True, figsize=(6,4))

    # Plot the waveform
    ax[0].plot(trace.times("matplotlib"), trace.data, rasterized=True, lw=0.6)
    #ax[0].set_ylim(-1e+2,1e+2)

    # First-order scattering coefficients
    ax[1].pcolormesh(timestamps, center_frequencies, order_1.T, rasterized=True)

    # Axes labels
    ax[1].set_yscale("log")
    #ax[0].set_ylabel("Counts")
    ax[1].set_ylabel("fc (Hz)", fontsize=12)
    
    ax[0].set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
    ax[1].set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
    
    
    for spine in ax[0].spines.values():
        spine.set_linewidth(1.5) 
    ax[0].tick_params(axis='both', which='major', length=4, width=1)  
    ax[0].tick_params(axis='both', which='minor', length=2, width=0.75)
    ax[0].tick_params(which='both', direction='out')
    
    for spine in ax[1].spines.values():
        spine.set_linewidth(1.5) 
    ax[1].tick_params(axis='both', which='major', length=4, width=1)  
    ax[1].tick_params(axis='both', which='minor', length=2, width=0.75)
    ax[1].tick_params(which='both', direction='out')

    # Show
    plt.show()



    center_frequencies = network_data.banks[1].centers



    # Create figure and axes
    fig, ax = plt.subplots(3, sharex=True, figsize=(6,4))

    # Plot the waveform
    ax[0].plot(trace.times("matplotlib"), trace.data, rasterized=True, lw=0.6)
    #ax[0].set_ylim(-1e+2,1e+2)
    
    ax[0].set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
    
    for spine in ax[0].spines.values():
        spine.set_linewidth(1.5) 
    ax[0].tick_params(axis='both', which='major', length=4, width=1)  
    ax[0].tick_params(axis='both', which='minor', length=2, width=0.75)
    ax[0].tick_params(which='both', direction='out')
        

    # Second-order scattering coefficients
    for i in range(1,3):
        order_2 = np.log10(scattering_coefficients[1][:, channel_id, :][:,i-1,:].squeeze())
        ax[i].pcolormesh(timestamps, center_frequencies, order_2.T, rasterized=True)
        
        # Axes labels
        ax[i].set_yscale("log")
        #ax[0].set_ylabel("Counts")
        ax[i].set_ylabel("fc (Hz)", fontsize=12)
        
        ax[i].set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
        
        for spine in ax[i].spines.values():
            spine.set_linewidth(1.5) 
        ax[i].tick_params(axis='both', which='major', length=4, width=1)  
        ax[i].tick_params(axis='both', which='minor', length=2, width=0.75)
        ax[i].tick_params(which='both', direction='out')
        

    # Show
    plt.show()
    
    
    Nseconds = int( (hdf5_endttime_jst-hdf5_starttime_jst).total_seconds() )
    
    
    np.savez(
        "example/scattering_coefficients"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".npz",
        order_1=scattering_coefficients[0],
        order_2=scattering_coefficients[1],
        times=timestamps,
    )
    
    
    """
    PSD
    """
    timestamps_spec = list()

    # Collect data and timestamps
    
    Fcur_arr = list()
    for traces in stream_scat[channel_id].slide(segment_duration,segment_duration):
        timestamps_spec.append(mdates.num2date(traces.times("matplotlib")[0]))
        traces_sub = traces.data #np.array([trace.data[:-1] for trace in traces])
        freqVec, Fcur = spectral_func.spec(traces_sub, sampRate=sampling_rate_hertz, percent_costaper=0.05)
        indd = np.where( (freqVec>=0.1) & (freqVec<=sampling_rate_hertz/2) )[0]
        freqVec = freqVec[indd]
        Fcur = Fcur[indd]
        #freqVec = freqVec[::10]
        #Fcur = Fcur[::10]
        Fcur_arr.append(Fcur)


    Fcur_arr = np.log10(np.array(Fcur_arr))


    fig = plt.figure(figsize=(10,3))
    ax = plt.subplot(111)
    amp = np.nanmedian(Fcur_arr)
    plt.pcolormesh(timestamps_spec, freqVec, Fcur_arr.T, rasterized=True, cmap=plt.cm.inferno)
    plt.yscale('log')
    plt.ylim(0.1,sampling_rate_hertz/2)
    plt.ylabel('Frequency [Hz]', fontsize=14)
    ax.set_xlim(timestamps_spec[0], timestamps_spec[-1]+datetime.timedelta(seconds=segment_duration_seconds))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1)) 
    plt.show()

