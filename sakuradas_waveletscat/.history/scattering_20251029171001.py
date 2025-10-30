import pickle
import os
import glob
import sys
import numpy as np
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
import pandas as pd
from obspy.core import UTCDateTime, Stream
import obspy
import datetime
#import spectral_func
from Params import *

sys.path.append("utils/")
from read_hdf5 import read_hdf5, read_hdf5_singlechannel
import spectral_func
from util_func import cosTaper

sys.path.append("scat/")
from network import ScatteringNetwork

try:
    import scienceplots
except:
    pass
plt.style.use(["science", "nature"])
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.linewidth"] = 1.0
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
plt.rcParams["axes.edgecolor"] = "#08192D"
plt.rcParams["axes.labelcolor"] = "#08192D"
plt.rcParams["xtick.color"] = "#08192D"
plt.rcParams["ytick.color"] = "#08192D"
plt.rcParams["text.color"] = "#08192D"
plt.rcParams["legend.framealpha"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["text.usetex"] = False
plt.rcParams["date.converter"] = "concise"


import sys
cptpath = r'get-cpt-master'
sys.path.append(cptpath)
import get_cpt

print(get_cpt.basedir) # the local repo
myurl_1 = 'get-cpt-master/cpt/rainbow.cpt'
mygmt_cmaps = get_cpt.get_cmap(myurl_1, method='list', N=256)




def plot_scattcoef_imshow(ustation, trace_x, trace_y):
    
    
    
    network_data = pickle.load(open("example/scattering_network.pickle", "rb"))
    center_f_1 =  network_data.banks[0].centers
    center_f_2 =  network_data.banks[1].centers
    print(center_f_1, center_f_2)
    
    
    
    scattering_coefficients = []
    times = []
    Nseconds = int( (hdf5_endttime_jst-hdf5_starttime_jst).total_seconds() )
    for _ in range(1):
        
        with np.load("example/scattering_coefficients"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".npz", allow_pickle=True) as data:
            order_1 = data["order_1"]
            order_2 = data["order_2"][:,0,:,:]
            times.extend( data["times"])

        
        # Reshape and stack scattering coefficients of all orders
        order_1 = order_1.reshape(order_1.shape[0], -1)[:,:]
        #order_2 = order_2[:,1,:].squeeze()[:,::-1]
        
        # order_2_vect = np.zeros((order_1.shape[0], int(order_1.shape[1]*order_2.shape[1])))
        for i in range(order_1.shape[1]):
            if i==0:
                order_2_vect = order_2[:,i,:].squeeze()[:,:]
            else:
                order_2_vect = np.hstack((order_2_vect, order_2[:,i,:].squeeze()[:,:]))
                
            
        
        #print(order_1.shape, order_2.shape, order_2_vect.shape)
        
        order_12 = np.hstack((order_1, order_2_vect))
        #print(order_12.shape)
        
        scattering_coefficients.extend( order_12 )
        
        #del order_1, order_2
        
    times = np.array(times)
    times_num = np.array( [ mdates.date2num(_) for _ in times] )
    scattering_coefficients = np.array(scattering_coefficients)
    print('taxis', times[0], times[-1])
    
    plt.figure(figsize=(12,10))

    
    # メインのイメージプロットとcolorbarを含むaxesの設定
    ax = plt.axes([0.1, 0.07, 0.8, 0.72])
    SC = plt.imshow(np.log10(scattering_coefficients).T, origin='upper',
            aspect='auto', vmin=-9, vmax=-5, cmap=mygmt_cmaps, interpolation='nearest',
            extent=(times[0]-datetime.timedelta(seconds=segment_duration_seconds*0.5), times[-1]+datetime.timedelta(seconds=segment_duration_seconds*0.5), 0, scattering_coefficients.shape[1], ), rasterized=True
            )
    cbar = plt.colorbar(SC, shrink=0.5)
    cbar.set_label(r'$\log_{10}$(wavelet scattering coef.)', fontsize=10)
    yticks_val = list(); yticks_label = list()
    idx = order_2.shape[2]*0.5
    for i in range(order_2.shape[1]):
        yticks_val.append(idx)
        yticks_label.append('2nd order '+str(int(order_2.shape[1]-i)).zfill(2))
        idx += order_2.shape[2]
    
    yticks_val.append(idx-order_2.shape[2]*0.5+order_1.shape[1]*0.5)
    yticks_label.append('1st order')
    
    
    plt.yticks(ticks=yticks_val, labels=yticks_label)
    
    ax.set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
    
    plt.gca().yaxis.set_minor_locator(plt.NullLocator())
    
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    """
    Hz ticks for 1st order
    """
    plt.text(times[-1]+datetime.timedelta(seconds=segment_duration_seconds*0.5), idx-order_2.shape[2]*0.5+1, '- '+format(center_f_1[-1], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
    plt.text(times[-1]+datetime.timedelta(seconds=segment_duration_seconds*0.5), idx-order_2.shape[2]*0.5+0.5+10, '- '+format(center_f_1[-10], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
    plt.text(times[-1]+datetime.timedelta(seconds=segment_duration_seconds*0.5), idx-order_2.shape[2]*0.5+0.5+19, '- '+format(center_f_1[0], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
    ax.axhline(y=idx-order_2.shape[2]*0.5, color='white', lw=1)
    
    """
    Hz ticks for 2nd order
    """
    tik_ini = 1.5
    for tik in range(len(center_f_1)):
        
        if tik==1:
            plt.text(times[-1]+datetime.timedelta(seconds=segment_duration_seconds*0.5), idx-tik_ini*order_2.shape[2]+0.5, '- '+format(center_f_2[-1], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
            plt.text(times[-1]+datetime.timedelta(seconds=segment_duration_seconds*0.5), idx-tik_ini*order_2.shape[2]+0.5+4, '- '+format(center_f_2[-5], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
            plt.text(times[-1]+datetime.timedelta(seconds=segment_duration_seconds*0.5), idx-tik_ini*order_2.shape[2]+0.5+9, '- '+format(center_f_2[0], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
            
        else:
            plt.text(times[-1]+datetime.timedelta(seconds=segment_duration_seconds*0.5), idx-tik_ini*order_2.shape[2]+0.5, '-', va='center', ha='left', color='k', fontsize=8)
            plt.text(times[-1]+datetime.timedelta(seconds=segment_duration_seconds*0.5), idx-tik_ini*order_2.shape[2]+0.5+4, '-', va='center', ha='left', color='k', fontsize=8)
            plt.text(times[-1]+datetime.timedelta(seconds=segment_duration_seconds*0.5), idx-tik_ini*order_2.shape[2]+0.5+9, '-', va='center', ha='left', color='k', fontsize=8)
            
        ax.axhline(y=idx-tik_ini*order_2.shape[2], color='white', lw=1)
            

        tik_ini += 1.0
        
        
    for spine in ax.spines.values():
        spine.set_linewidth(1.5) 
    ax.tick_params(axis='both', which='major', length=4, width=1)  
    ax.tick_params(axis='both', which='minor', length=2, width=0.75)
    ax.tick_params(which='both', direction='out')
    
    
    # colorbarを含めた全体の幅に合わせてax0も同じ位置と幅に設定
    pos = ax.get_position()
    ax0 = plt.axes([pos.x0, 0.85, pos.width, 0.1])
    ax0.plot(trace_x, trace_y)
    
    ax0.set_ylim(-1,1)
    
    
    
    ax0.set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
    
    for spine in ax0.spines.values():
        spine.set_linewidth(1.5) 
    ax0.tick_params(axis='both', which='major', length=4, width=1)  
    ax0.tick_params(axis='both', which='minor', length=2, width=0.75)
    ax0.tick_params(which='both', direction='out')
    
    plt.suptitle(ustation, fontsize=14, x=pos.x0+0.5*pos.width)
    
    plt.savefig("Figure/"+fiber+"/"+hdf5_starttime_jst.strftime("%Y")+"/"+hdf5_starttime_jst.strftime("%m")+"/"+hdf5_starttime_jst.strftime("%d")+"/"+hdf5_starttime_jst.strftime("%H")+"/scattering_coefficients_tchange_"+ustation+"_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".png", dpi=300, bbox_inches="tight")
    plt.close()
    
    


if __name__ == "__main__":
    
    network_data = pickle.load(open("example/scattering_network.pickle", "rb"))
    
    
    
    
    
    for ch_idx, target_channel in enumerate(used_channel_list):
        
        
        
        hdf5_starttime_utc = hdf5_starttime_jst + datetime.timedelta(hours=-9)
        hdf5_file_list = []
        for mm in range(N_minute):
            ts_utc = hdf5_starttime_utc + datetime.timedelta(minutes=mm)
            
            hdf5_dirname = "/Users/hirosetakashi/Volumes/noise_monitoring/noise_monitoring/DAS/Tohoku_15/Fiber-2_HDF5/"+ts_utc.strftime("%Y")+"/"+ts_utc.strftime("%m")+"/"+ts_utc.strftime("%d")+"/" 
            
            print( hdf5_dirname+"decimator_"+ts_utc.strftime("%Y-%m-%d_%H.%M.%S")+"_UTC_"+"*.h5")
            filename = glob.glob(
                hdf5_dirname+"decimator_"+ts_utc.strftime("%Y-%m-%d_%H.%M.%S")+"_UTC_"+"*.h5"
            )[0] 
            
            print(filename)
            
            hdf5_file_list.append(filename)
            
        stream_minute = Stream()
        for i in range(len(hdf5_file_list)):
            stream_minute += read_hdf5_singlechannel(hdf5_file_list[i], fiber, int(used_channel_list[ch_idx]))
            
        stream_minute.merge(method=1)
        
        for tr in stream_minute:
            if np.ma.is_masked(tr.data):
                tr.data = tr.data.filled(0)  
        
        stream_minute.resample(Fs, no_filter=False, window="hann")
        
            
        stream_scat = stream_minute.select(station=used_channel_list[ch_idx])
        
        stream_scat.write('trace.sac', format='sac')  
        
        print(stream_scat[0].stats)
        ustation = stream_scat[0].stats.network.lower() + used_channel_list[ch_idx]
        
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
                traces_sub = np.concatenate((traces_sub, np.zeros((Nch,padd))), axis=1)
            
            traces_sub *= tp
            
            
            
            segments.append(traces_sub)
        
        
        scattering_coefficients = network_data.transform(segments, reduce_type=np.max)
        
        # Extract the first channel
        channel_id = 0
        trace = stream_scat[channel_id]
        

        order_1 = np.log10(scattering_coefficients[0][:, channel_id, :].squeeze())
        center_frequencies = network_data.banks[0].centers


        # # Create figure and axes
        # fig, ax = plt.subplots(2, sharex=True, figsize=(6,4))

        # # Plot the waveform
        # ax[0].plot(trace.times("matplotlib"), trace.data/np.nanmax(np.abs(trace.data)), rasterized=True, lw=0.6)
        # ax[0].set_ylim(-1,1)
        

        # # First-order scattering coefficients
        # ax[1].pcolormesh(timestamps, center_frequencies, order_1.T, rasterized=True)

        # # Axes labels
        # ax[1].set_yscale("log")
        # #ax[0].set_ylabel("Counts")
        # ax[1].set_ylabel("fc (Hz)", fontsize=12)
        
        # ax[0].set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
        # ax[1].set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
        
        
        # for spine in ax[0].spines.values():
        #     spine.set_linewidth(1.5) 
        # ax[0].tick_params(axis='both', which='major', length=4, width=1)  
        # ax[0].tick_params(axis='both', which='minor', length=2, width=0.75)
        # ax[0].tick_params(which='both', direction='out')
        
        # for spine in ax[1].spines.values():
        #     spine.set_linewidth(1.5) 
        # ax[1].tick_params(axis='both', which='major', length=4, width=1)  
        # ax[1].tick_params(axis='both', which='minor', length=2, width=0.75)
        # ax[1].tick_params(which='both', direction='out')

        # # Show
        # plt.suptitle(stream_scat[0].stats.network.lower()+stream_scat[0].stats.station+" "+stream_scat[0].stats.starttime.strftime("%Y-%m-%d %H:%M:%S")+"-"+stream_scat[0].stats.endtime.strftime("%Y-%m-%d %H:%M:%S"), fontsize=12)
        
        # os.makedirs("Figure/", exist_ok=True)
        # #plt.savefig("Figure/scattering_coefficients_1st_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".png", dpi=300, bbox_inches="tight")
        # plt.show()



        center_frequencies = network_data.banks[1].centers



        # # Create figure and axes
        # fig, ax = plt.subplots(3, sharex=True, figsize=(6,4))

        # # Plot the waveform
        # ax[0].plot(trace.times("matplotlib"), trace.data/np.nanmax(np.abs(trace.data)), rasterized=True, lw=0.6)
        # ax[0].set_ylim(-1,1)
        
        # ax[0].set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
        
        # for spine in ax[0].spines.values():
        #     spine.set_linewidth(1.5) 
        # ax[0].tick_params(axis='both', which='major', length=4, width=1)  
        # ax[0].tick_params(axis='both', which='minor', length=2, width=0.75)
        # ax[0].tick_params(which='both', direction='out')
            

        # # Second-order scattering coefficients
        # for i in range(1,3):
        #     order_2 = np.log10(scattering_coefficients[1][:, channel_id, :][:,i-1,:].squeeze())
        #     ax[i].pcolormesh(timestamps, center_frequencies, order_2.T, rasterized=True)
            
        #     # Axes labels
        #     ax[i].set_yscale("log")
        #     #ax[0].set_ylabel("Counts")
        #     ax[i].set_ylabel("fc (Hz)", fontsize=12)
            
        #     ax[i].set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
            
        #     for spine in ax[i].spines.values():
        #         spine.set_linewidth(1.5) 
        #     ax[i].tick_params(axis='both', which='major', length=4, width=1)  
        #     ax[i].tick_params(axis='both', which='minor', length=2, width=0.75)
        #     ax[i].tick_params(which='both', direction='out')
            

        # # Show
        # plt.suptitle(stream_scat[0].stats.network.lower()+stream_scat[0].stats.station+" "+stream_scat[0].stats.starttime.strftime("%Y-%m-%d %H:%M:%S")+"-"+stream_scat[0].stats.endtime.strftime("%Y-%m-%d %H:%M:%S"), fontsize=12)
        # plt.show()
        
        
        
        
        
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
            timestamps_spec.append(mdates.num2date(traces.times("matplotlib")[0])+datetime.timedelta(seconds=segment_duration_seconds*0.5))
            print(mdates.num2date(traces.times("matplotlib")[0]))
            traces_sub = traces.data #np.array([trace.data[:-1] for trace in traces])
            freqVec, Fcur = spectral_func.spec(traces_sub, sampRate=sampling_rate_hertz, percent_costaper=0.05)
            indd = np.where( (freqVec>=0.1) & (freqVec<=sampling_rate_hertz/2) )[0]
            freqVec = freqVec[indd]
            Fcur = Fcur[indd]
            #freqVec = freqVec[::10]
            #Fcur = Fcur[::10]
            Fcur_arr.append(Fcur)


        Fcur_arr = np.log10(np.array(Fcur_arr))
        
        
        fig, ax = plt.subplots(2, sharex=True, figsize=(6,4))

        # Plot the waveform
        ax[0].plot(trace.times("matplotlib"), trace.data/np.nanmax(np.abs(trace.data)), rasterized=True, lw=0.6)
        ax[0].set_ylim(-1,1)
        
        
        # First-order scattering coefficients
        
        ax[1].pcolormesh(timestamps_spec, freqVec, Fcur_arr.T, rasterized=True, cmap=plt.cm.inferno)
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
        plt.suptitle(stream_scat[0].stats.network.lower()+stream_scat[0].stats.station+" "+stream_scat[0].stats.starttime.strftime("%Y-%m-%d %H:%M:%S")+"-"+stream_scat[0].stats.endtime.strftime("%Y-%m-%d %H:%M:%S"), fontsize=12)
        
        os.makedirs("Figure/"+fiber+"/"+hdf5_starttime_jst.strftime("%Y")+"/"+hdf5_starttime_jst.strftime("%m")+"/"+hdf5_starttime_jst.strftime("%d")+"/"+hdf5_starttime_jst.strftime("%H"), exist_ok=True)
        plt.savefig("Figure/"+fiber+"/"+hdf5_starttime_jst.strftime("%Y")+"/"+hdf5_starttime_jst.strftime("%m")+"/"+hdf5_starttime_jst.strftime("%d")+"/"+hdf5_starttime_jst.strftime("%H")+"/psd_"+ustation+"_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".png", dpi=300, bbox_inches="tight")
        plt.close()

        
        trace_x = trace.times("matplotlib")
        trace_y = trace.data/np.nanmax(np.abs(trace.data))
        
        plot_scattcoef_imshow(ustation, trace_x, trace_y)