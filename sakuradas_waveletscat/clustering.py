import pickle
import os
import glob
import copy
import sys
import numpy as np
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA
from obspy.core import UTCDateTime, Stream
import obspy
import datetime
from sklearn.cluster import KMeans
import datetime
import fastcluster
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster import hierarchy
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


def cluster_and_plot_dendrogram( Z, threshold, default_color='black', pooling='max', savefig=True):

    # get cluster labels
    labels         = hierarchy.fcluster(Z, threshold, criterion='distance')-1
    labels_str     = [f"cluster {l+1}: n={c}\n" for (l,c) in zip(*np.unique(labels, return_counts=True))]
    n_clusters     = len(labels_str)
    
    colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
    c01 = []
    for i in range(n_clusters+1):
        c01.append( colors[np.mod(i,len(colors))] )

    cluster_colors = [c for c in c01]
    cluster_colors_array = [cluster_colors[l] for l in labels]
    
    link_cols = {}
    for i, i12 in enumerate(Z[:,:2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(Z) else cluster_colors_array[x] for x in i12)
        link_cols[i+1+len(Z)] = c1 if c1 == c2 else 'k'
        
    
    
    # plot dendrogram with colored clusters
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot(111)
    
    #plt.xlabel('Data points', fontsize=12)
    plt.ylabel('Distance', fontsize=12)

    # plot dendrogram based on clustering results
    dendro = hierarchy.dendrogram(
        Z,
        labels = labels,
        color_threshold=threshold,
        truncate_mode = 'lastp',
        p = n_clusters,
        show_leaf_counts = True,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=False,
        link_color_func=lambda x: link_cols[x],
        above_threshold_color=default_color,
        ax=plt.gca()
    )
    #print(dendro)
    
    leaf_colors = dendro['leaves_color_list']
    leaf_ivls = dendro['ivl']
    leaf_ivls_label = []
    for i, _ in enumerate(leaf_ivls):
        leaf_ivls_label.append('cluster '+str(i+1)+' '+str(_))
    
    plt.axhline(threshold, color='gray', ls='--')
    
    # for i, s in enumerate(labels_str):
    #     plt.text(1.05, 0.95-i*0.04, s,
    #             transform=plt.gca().transAxes,
    #             va='top', ha='left', color=cluster_colors[i], fontsize=10)
        
    # 葉のラベルを取得
    leaf_labels = plt.gca().get_xticklabels()
    # 葉のx座標を計算
    leaf_x_coordinates = [label.get_position()[0] for label in leaf_labels]
    
    # print(ax.get_xticks())
    # print([label.get_position() for label in leaf_labels])
    # print(ax.get_ylim())
    # x座標を使用してテキストをプロット
    for i, x in enumerate(leaf_x_coordinates):
        plt.text(x, -1, leaf_ivls_label[i], va='top', ha='center', color=c01[i], fontsize=14, rotation=90, backgroundcolor='white')
        
    
    fig.patch.set_facecolor('white')
    plt.minorticks_off()
    
    plt.tight_layout()
    
    # if savefig:
    #     if pooling=='average':
    #         os.makedirs(figout_dirname+'_averagepooling', exist_ok=True)
    #         plt.savefig(figout_dirname+'_averagepooling/dendrogram.pdf', dpi=200)
    #     else:
    #         os.makedirs(figout_dirname, exist_ok=True)
    #         plt.savefig(figout_dirname+'/dendrogram.pdf', dpi=200)
    
    plt.show()

def clustering(threshold, pooling='max', savefig=False):
    
    
    with np.load("example/independent_components_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".npz", allow_pickle=True) as data:
        features = data["features"]
        times = data["times"]
    
    print(features.shape)

    Z = fastcluster.linkage(features, method='ward', metric='euclidean', preserve_input='True') 
    
    cluster_and_plot_dendrogram(copy.deepcopy(Z), threshold, default_color='black', pooling=pooling, savefig=savefig)
    
    
    np.save("dendrogram_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+'.npy', Z)

if __name__ == "__main__":
    
    scattering_coefficients = []
    times = []
    
    

    # Load data from file
    with np.load("example/scattering_coefficients"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".npz", allow_pickle=True) as data:
        order_1 = data["order_1"]
        order_2 = data["order_2"]
        times.extend( data["times"])

    # Reshape and stack scattering coefficients of all orders
    order_1 = order_1.reshape(order_1.shape[0], -1)
    order_2 = order_2.reshape(order_2.shape[0], -1)
    #print(order_1.shape)
    #print(order_2.shape)
    scattering_coefficients.extend( np.hstack((order_1, order_2)) )
    #scattering_coefficients = order_1
        
    times = np.array(times)
    scattering_coefficients = np.array(scattering_coefficients)
    print(scattering_coefficients.shape)

    # transform into log
    scattering_coefficients = np.log(scattering_coefficients)

    # print info about shape
    n_times, n_coeff = scattering_coefficients.shape
    print("Collected {} samples of {} dimensions each.".format(n_times, n_coeff))

    model = FastICA(n_components=5, whiten="unit-variance", random_state=42)
    #model = FastICA(n_components=10, whiten="unit-variance", random_state=42)
    #model = SparsePCA(n_components=10, random_state=0, ridge_alpha=1.0)
    features = model.fit_transform(scattering_coefficients)
    
    
    # Normalize features for display
    features_normalized = features / np.abs(features).max(axis=0)
    
    
    # Save the features
    np.savez(
        "example/independent_components_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".npz",
        features=features,
        times=times,
    )


    # Save the dimension reduction model
    with open("example/dimension_model_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".pickle", "wb") as pickle_file:
        pickle.dump(
            model,
            pickle_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    
    
    
    # Load features and datetimes from file
    with np.load("example/independent_components_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".npz", allow_pickle=True) as data:
        features = data["features"]
        times = data["times"]

    # Load network
    network = pickle.load(open("example/scattering_network.pickle", "rb"))

    print(times)
    
    Z = fastcluster.linkage(features, method='ward', metric='euclidean', preserve_input='True')
    
    threshold = 8
    predictions = fcluster(Z, threshold, criterion="distance")
    clustering(threshold, pooling='max', savefig=False)
    
    
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
    
    
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax1 = plt.subplot(211)
    ax1.plot(np.arange(0, windL, 1/Fs), stream_scat[0].data, color='black', lw=0.5)
    ax1.set_xlim(0, windL)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5) 
    ax1.tick_params(axis='both', which='major', length=4, width=1)  
    ax1.tick_params(axis='both', which='minor', length=2, width=0.75)
    ax1.tick_params(which='both', direction='out')
    
    ax2 = plt.subplot(212)
    times_lapset = np.array( [ (times[_]-times[0]).total_seconds() for _ in range(len(times)) ] )
    time_step = 0.5*(times_lapset[1]-times_lapset[0])
    times_lapset += time_step
    print(times_lapset)
    ax2.scatter(times_lapset, predictions)
    ax2.set_xlim(0, windL)
    ax2.set_ylim(predictions.min()-1, predictions.max()+1)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5) 
    ax2.tick_params(axis='both', which='major', length=4, width=1)  
    ax2.tick_params(axis='both', which='minor', length=2, width=0.75)
    ax2.tick_params(which='both', direction='out')
    
    plt.show()
    
    

    

