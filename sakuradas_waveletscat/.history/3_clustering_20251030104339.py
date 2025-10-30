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
from read_hdf5 import read_hdf5, read_hdf5_singlechannel

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


def cluster_and_plot_dendrogram( ustation, Z, threshold, default_color='black', pooling='max', savefig=True):

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
    plt.title(ustation, fontsize=12)
    plt.tight_layout()
    
    # if savefig:
    #     if pooling=='average':
    #         os.makedirs(figout_dirname+'_averagepooling', exist_ok=True)
    #         plt.savefig(figout_dirname+'_averagepooling/dendrogram.pdf', dpi=200)
    #     else:
    #         os.makedirs(figout_dirname, exist_ok=True)
    #         plt.savefig(figout_dirname+'/dendrogram.pdf', dpi=200)
    
    print("save dendrogram figure", "Figure/"+fiber+"/"+hdf5_starttime_jst.strftime("%Y")+"/"+hdf5_starttime_jst.strftime("%m")+"/"+hdf5_starttime_jst.strftime("%d")+"/"+hdf5_starttime_jst.strftime("%H")+"/dendrogram_"+ustation+"_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".png")

    plt.savefig("Figure/"+fiber+"/"+hdf5_starttime_jst.strftime("%Y")+"/"+hdf5_starttime_jst.strftime("%m")+"/"+hdf5_starttime_jst.strftime("%d")+"/"+hdf5_starttime_jst.strftime("%H")+"/dendrogram_"+ustation+"_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".png", dpi=300, bbox_inches="tight")
    plt.close()

def clustering(ustation, threshold, pooling='max', savefig=False):
    
    
    with np.load("example/independent_components_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".npz", allow_pickle=True) as data:
        features = data["features"]
        times = data["times"]
    
    print(features.shape)

    Z = fastcluster.linkage(features, method='ward', metric='euclidean', preserve_input='True') 
    
    cluster_and_plot_dendrogram(ustation, copy.deepcopy(Z), threshold, default_color='black', pooling=pooling, savefig=savefig)
    
    print("save dendrogram data", "dendrogram_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+'.npy')
    np.save("example/dendrogram_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+'.npy', Z)

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
    print("save independent components", "example/independent_components_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".npz")
    np.savez(
        "example/independent_components_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".npz",
        features=features,
        times=times,
    )


    # Save the dimension reduction model
    print("save dimension model", "example/dimension_model_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".pickle")
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

    
    
    Z = fastcluster.linkage(features, method='ward', metric='euclidean', preserve_input='True')
    
    
    predictions = fcluster(Z, threshold, criterion="distance")
    
    print(predictions)
    
    Nclusters = predictions.max()
    
    
    
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
    
    
    scattering_coefficients = np.array(scattering_coefficients)
    
    scattering_coefficients_vals_plot = np.zeros((scattering_coefficients.shape[1], Nclusters+1)) * np.nan
    for cluster_idx in range(1,Nclusters+1): #Nclusters+2
        
        hit_idx = np.where(predictions == cluster_idx)[0]
        
        scattering_coefficients_vals_plot[:,cluster_idx-1] = np.nanmedian(scattering_coefficients[hit_idx,:], axis=0)
    
    
    
    network_data = pickle.load(open("example/scattering_network.pickle", "rb"))
    center_f_1 =  network_data.banks[0].centers
    center_f_2 =  network_data.banks[1].centers
    
    plt.figure(figsize=(6,8))
    ax = plt.subplot(111)
    
    SC = ax.imshow(np.log10(scattering_coefficients_vals_plot), origin='upper', extent=(0.5, Nclusters+1.5, 0, scattering_coefficients.shape[1], ), 
            aspect='auto', vmin=-9, vmax=-5, cmap=mygmt_cmaps, interpolation='nearest', rasterized=True)
    cbar = plt.colorbar(SC, shrink=0.5, pad=0.15)
    cbar.set_label(r'median $\log_{10}$(wavelet scattering coef.)', fontsize=10)
    
    plt.xticks(list(range(1,Nclusters+2)))
    yticks_val = list(); yticks_label = list()
    idx = order_2.shape[2]*0.5
    for i in range(order_2.shape[1]):
        yticks_val.append(idx)
        yticks_label.append('2nd order '+str(int(order_2.shape[1]-i)).zfill(2))
        idx += order_2.shape[2]
    
    yticks_val.append(idx-order_2.shape[2]*0.5+order_1.shape[1]*0.5)
    yticks_label.append('1st order')
    plt.yticks(ticks=yticks_val, labels=yticks_label)
    
    """
    Hz ticks for 1st order
    """
    plt.text(Nclusters+0.5, idx-order_2.shape[2]*0.5+1, '- '+format(center_f_1[-1], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
    plt.text(Nclusters+0.5, idx-order_2.shape[2]*0.5+0.5+10, '- '+format(center_f_1[-10], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
    plt.text(Nclusters+0.5, idx-order_2.shape[2]*0.5+0.5+19, '- '+format(center_f_1[0], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
    ax.axhline(y=idx-order_2.shape[2]*0.5, color='white', lw=1)
    
    """
    Hz ticks for 2nd order
    """
    tik_ini = 1.5
    for tik in range(len(center_f_1)):
        
        if tik==1:
            plt.text(Nclusters+0.5, idx-tik_ini*order_2.shape[2]+0.5, '- '+format(center_f_2[-1], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
            plt.text(Nclusters+0.5, idx-tik_ini*order_2.shape[2]+0.5+4, '- '+format(center_f_2[-5], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
            plt.text(Nclusters+0.5, idx-tik_ini*order_2.shape[2]+0.5+9, '- '+format(center_f_2[0], ".2f")+' Hz', va='center', ha='left', color='k', fontsize=8)
            
        else:
            plt.text(Nclusters+0.5, idx-tik_ini*order_2.shape[2]+0.5, '-', va='center', ha='left', color='k', fontsize=8)
            plt.text(Nclusters+0.5, idx-tik_ini*order_2.shape[2]+0.5+4, '-', va='center', ha='left', color='k', fontsize=8)
            plt.text(Nclusters+0.5, idx-tik_ini*order_2.shape[2]+0.5+9, '-', va='center', ha='left', color='k', fontsize=8)
            
        ax.axhline(y=idx-tik_ini*order_2.shape[2], color='white', lw=1)
            

        tik_ini += 1.0
    
    
    plt.xlim(0.5, Nclusters+0.5)
    
    plt.xlabel('cluster', fontsize=12)
    plt.minorticks_off()
    for i in range(1,Nclusters+1):
        ax.axvline(x=i+0.5, color='white', lw=1)
        
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ustation = fiber[0:3] + used_channel_list[0]
    plt.suptitle(ustation, fontsize=12)
    
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5) 
    ax.tick_params(axis='both', which='major', length=4, width=1)  
    ax.tick_params(axis='both', which='minor', length=2, width=0.75)
    ax.tick_params(which='both', direction='out')

    
    
    plt.tight_layout()

    print("save scattering figure", "Figure/"+fiber+"/"+hdf5_starttime_jst.strftime("%Y")+"/"+hdf5_starttime_jst.strftime("%m")+"/"+hdf5_starttime_jst.strftime("%d")+"/"+hdf5_starttime_jst.strftime("%H")+"/scattering_coefficients_allclusters_"+ustation+"_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".png")
    
    plt.savefig("Figure/"+fiber+"/"+hdf5_starttime_jst.strftime("%Y")+"/"+hdf5_starttime_jst.strftime("%m")+"/"+hdf5_starttime_jst.strftime("%d")+"/"+hdf5_starttime_jst.strftime("%H")+"/scattering_coefficients_allclusters_"+ustation+"_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".png", dpi=300, bbox_inches="tight")
    plt.close()
    
    
    ustation = 'noj'+used_channel_list[0]
    clustering(ustation, threshold, pooling='max', savefig=True)
    
    
    
    stream_minute = obspy.read('trace.sac')  
    
    print(stream_minute)

    stream_scat = stream_minute.select(station=used_channel_list[0])
    ustation = fiber[0:3] + used_channel_list[0]
    
    print('stream_scat[0].data', stream_scat[0].data)

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    
    
    trace = stream_scat[0]
        
    print(trace.times("matplotlib"))
    ax1 = plt.subplot(211)
    ax1.plot(trace.times("matplotlib"), stream_scat[0].data/np.nanmax(np.abs(stream_scat[0].data)), color='black', lw=0.5)
    ax1.set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
    ax1.set_ylabel('strain', fontsize=12)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5) 
    ax1.tick_params(axis='both', which='major', length=4, width=1)  
    ax1.tick_params(axis='both', which='minor', length=2, width=0.75)
    ax1.tick_params(which='both', direction='out')
    ### major ticksを3時間ごと， minor ticksを1時間ごとに設定
    ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24), interval=3))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0,24), interval=1))

    
    
    ax2 = plt.subplot(212)
    colors = [f'C{i-1}' for i in predictions]  
    ax2.scatter(times, predictions, c=colors)
    
    unique_predictions = np.unique(predictions)
    for cluster_id in unique_predictions:
        ax2.scatter([], [], c=f'C{cluster_id-1}', label=f'Cluster {cluster_id}')
    #ax2.legend(loc='upper right', frameon=True)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)
    ax2.set_ylabel('cluster', fontsize=12)
    ax2.set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
    ax2.set_ylim(predictions.min()-1, predictions.max()+1)
    
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5) 
    ax2.tick_params(axis='both', which='major', length=4, width=1)  
    ax2.tick_params(axis='both', which='minor', length=2, width=0.75)
    ax2.tick_params(which='both', direction='out')
    ### major ticksを3時間ごと， minor ticksを1時間ごとに設定
    ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24), interval=3))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0,24), interval=1))
    
    
    
    plt.suptitle(ustation, fontsize=12)

    print("save clustering figure", "Figure/"+fiber+"/"+hdf5_starttime_jst.strftime("%Y")+"/"+hdf5_starttime_jst.strftime("%m")+"/"+hdf5_starttime_jst.strftime("%d")+"/"+hdf5_starttime_jst.strftime("%H")+"/clustering_"+ustation+"_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".png")
    
    plt.savefig("Figure/"+fiber+"/"+hdf5_starttime_jst.strftime("%Y")+"/"+hdf5_starttime_jst.strftime("%m")+"/"+hdf5_starttime_jst.strftime("%d")+"/"+hdf5_starttime_jst.strftime("%H")+"/clustering_"+ustation+"_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".png", dpi=300, bbox_inches="tight")
    plt.close()





