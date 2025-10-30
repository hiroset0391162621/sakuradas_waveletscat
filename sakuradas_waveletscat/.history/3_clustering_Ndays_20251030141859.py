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


def cluster_and_plot_dendrogram( ustation, Z, threshold, default_color='black', pooling='max', savefig=True, output_dir=None, filename_suffix=None):

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
    
    if output_dir is None or filename_suffix is None:
        # single-day legacy path
        save_path = "Figure/"+fiber+"/"+hdf5_starttime_jst.strftime("%Y")+"/"+hdf5_starttime_jst.strftime("%m")+"/"+hdf5_starttime_jst.strftime("%d")+"/"+hdf5_starttime_jst.strftime("%H")+"/dendrogram_"+ustation+"_"+hdf5_starttime_jst.strftime("%Y%m%d%H%M")+"_"+str(Nseconds)+".png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        save_path = f"{output_dir}dendrogram_{ustation}_{filename_suffix}.png"

    print("save dendrogram figure", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
    
    # ==========================================
    # 複数日のデータを読み込むための設定
    # ==========================================
    start_date = datetime.datetime(2025, 5, 15, 0, 0, 0)  # 開始日時
    end_date = datetime.datetime(2025, 5, 20, 0, 0, 0)    # 終了日時
    
    scattering_coefficients = []
    times = []
    
    # 各日のデータを順次読み込む
    current_date = start_date
    while current_date < end_date:
        next_date = current_date + datetime.timedelta(days=1)
        if next_date > end_date:
            next_date = end_date
        
        Nseconds_day = int((next_date - current_date).total_seconds())
        filename = f"example/scattering_coefficients{current_date.strftime('%Y%m%d%H%M')}_{Nseconds_day}.npz"
        
        # ファイルが存在するかチェック
        if os.path.exists(filename):
            print(f"Loading: {filename}")
            try:
                with np.load(filename, allow_pickle=True) as data:
                    order_1_day = data["order_1"]
                    order_2_day = data["order_2"]
                    times_day = data["times"]
                
                # Reshape and stack scattering coefficients of all orders
                order_1_day = order_1_day.reshape(order_1_day.shape[0], -1)
                order_2_day = order_2_day.reshape(order_2_day.shape[0], -1)
                
                scattering_coefficients.extend(np.hstack((order_1_day, order_2_day)))
                times.extend(times_day)
                
                print(f"  Loaded {order_1_day.shape[0]} samples")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        else:
            print(f"File not found: {filename}")
        
        current_date = next_date
    
    # Convert to numpy arrays
    times = np.array(times)
    scattering_coefficients = np.array(scattering_coefficients)
    print(f"\nTotal scattering coefficients shape: {scattering_coefficients.shape}")
    print(f"Date range: {times[0]} to {times[-1]}")

    # ファイル名用の文字列を生成
    date_range_str = f"{start_date.strftime('%Y%m%d%H%M')}_{end_date.strftime('%Y%m%d%H%M')}"
    total_seconds = int((end_date - start_date).total_seconds())
    
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
    print("save independent components", f"example/independent_components_{date_range_str}.npz")
    np.savez(
        f"example/independent_components_{date_range_str}.npz",
        features=features,
        times=times,
    )


    # Save the dimension reduction model
    print("save dimension model", f"example/dimension_model_{date_range_str}.pickle")
    with open(f"example/dimension_model_{date_range_str}.pickle", "wb") as pickle_file:
        pickle.dump(
            model,
            pickle_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    
    
    
    # Load features and datetimes from file (use the saved multi-day data)
    with np.load(f"example/independent_components_{date_range_str}.npz", allow_pickle=True) as data:
        features = data["features"]
        times = data["times"]

    # Load network
    network = pickle.load(open("example/scattering_network.pickle", "rb"))

    
    
    Z = fastcluster.linkage(features, method='ward', metric='euclidean', preserve_input='True')
    
    # Save dendrogram data for multi-day
    dendrogram_filename = f"example/dendrogram_{date_range_str}.npy"
    print(f"save dendrogram data: {dendrogram_filename}")
    np.save(dendrogram_filename, Z)
    
    # Plot and save dendrogram figure for multi-day
    ustation_plot = fiber[0:3] + used_channel_list[0]
    output_dir = f"Figure/{fiber}/multiday_{date_range_str}/"
    cluster_and_plot_dendrogram(ustation_plot, copy.deepcopy(Z), threshold, default_color='black', pooling=pooling, savefig=True, output_dir=output_dir, filename_suffix=date_range_str)
    
    predictions = fcluster(Z, threshold, criterion="distance")
    
    print(predictions)
    
    Nclusters = predictions.max()
    print(f"Number of clusters: {Nclusters}")
    
    
    
    # 複数日のscattering coefficientsを再度読み込む（プロット用）
    scattering_coefficients = []
    times_plot = []
    order_1_all = []  # collect raw 1st-order coefficients for per-cluster spectra
    
    current_date = start_date
    while current_date < end_date:
        next_date = current_date + datetime.timedelta(days=1)
        if next_date > end_date:
            next_date = end_date
        
        Nseconds_day = int((next_date - current_date).total_seconds())
        filename = f"example/scattering_coefficients{current_date.strftime('%Y%m%d%H%M')}_{Nseconds_day}.npz"
        
        if os.path.exists(filename):
            print(f"Loading for plot: {filename}")
            try:
                with np.load(filename, allow_pickle=True) as data:
                    order_1_day_raw = data["order_1"]
                    order_2 = data["order_2"][:,0,:,:]
                    times_plot.extend(data["times"])

                # Reshape and stack scattering coefficients of all orders
                order_1 = order_1_day_raw.reshape(order_1_day_raw.shape[0], -1)[:,:]
                
                for i in range(order_1.shape[1]):
                    if i==0:
                        order_2_vect = order_2[:,i,:].squeeze()[:,:]
                    else:
                        order_2_vect = np.hstack((order_2_vect, order_2[:,i,:].squeeze()[:,:]))
                
                order_12 = np.hstack((order_1, order_2_vect))
                scattering_coefficients.extend(order_12)
                order_1_all.append(order_1_day_raw)
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        
        current_date = next_date
    
    scattering_coefficients = np.array(scattering_coefficients)
    times_plot = np.array(times_plot)
    order_1_all = np.concatenate(order_1_all, axis=0) if len(order_1_all) > 0 else None
    
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

    # 複数日の場合の保存先ディレクトリ
    output_dir = f"Figure/{fiber}/multiday_{date_range_str}/"
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{output_dir}scattering_coefficients_allclusters_{ustation}_{date_range_str}.png"
    print(f"save scattering figure: {output_filename}")
    
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------
    # Plot median 1st-order scattering spectra per cluster (overlay)
    # ---------------------------------------------
    try:
        if (order_1_all is not None) and (order_1_all.shape[0] == len(predictions)):
            # reshape to (N, F1, -1) and reduce trailing dims
            o1 = order_1_all.reshape(order_1_all.shape[0], order_1_all.shape[1], -1)
            o1_reduced = np.nanmedian(o1, axis=2)  # (N, F1)

            fig = plt.figure(figsize=(8, 6))
            ax = plt.subplot(111)
            eps = 1e-12
            for cid in np.unique(predictions):
                idx = np.where(predictions == cid)[0]
                if idx.size == 0:
                    continue
                spec_med = np.nanmedian(o1_reduced[idx, :], axis=0)
                print(f"Cluster {cid}: {spec_med}")
                fvec = np.asarray(center_f_1)
                m = min(len(spec_med), len(fvec))
                ax.plot(fvec[:m], np.log10(spec_med[:m] + eps), lw=1.5, label=f"Cluster {cid}")

            ax.set_xscale('log')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('median log10(1st-order scattering coef.)')
            ax.grid(True, which='both', ls=':', alpha=0.5)
            ax.legend(loc='best', fontsize=9, ncols=2)
            fig.tight_layout()

            out_spec = f"{output_dir}order1_spectra_overlay_{ustation}_{date_range_str}.png"
            print(f"save 1st-order spectra overlay: {out_spec}")
            fig.savefig(out_spec, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            print("Skip 1st-order spectra overlay: order_1_all missing or length mismatch.")
    except Exception as e:
        print(f"Error while plotting 1st-order spectra overlay: {e}")
    
    
    ustation = 'noj'+used_channel_list[0]
    # clustering(ustation, threshold, pooling='max', savefig=True)  # 複数日対応が必要な場合は後で修正
    
    
    
    # 複数日のSACファイルを読み込んで結合
    stream_minute_all = Stream()
    current_date = start_date
    while current_date < end_date:
        next_date = current_date + datetime.timedelta(days=1)
        if next_date > end_date:
            next_date = end_date

        Nseconds_day = int((next_date - current_date).total_seconds())
        readsac_fname = f"sac/{ustation}_{current_date.strftime('%Y%m%d%H%M')}_{Nseconds_day}.sac"

        if os.path.exists(readsac_fname):
            try:
                st_day = obspy.read(readsac_fname)
                stream_minute_all += st_day
                print(f"Loaded SAC: {readsac_fname}")
            except Exception as e:
                print(f"Error reading {readsac_fname}: {e}")
        else:
            print(f"SAC not found: {readsac_fname}")

        current_date = next_date

    if len(stream_minute_all) == 0:
        print("No SAC files loaded for the selected date range. Skipping waveform plot.")

    # 対象ステーションでフィルタして結合
    stream_scat = stream_minute_all.select(station=used_channel_list[0])
    try:
        stream_scat.merge(method=1, fill_value=0)
    except Exception as e:
        print(f"Merge warning (continuing without merge): {e}")
    ustation = fiber[0:3] + used_channel_list[0]
    
    if len(stream_scat) == 0:
        print("No SAC traces for the target station. Skipping waveform plot.")
    else:
        print('stream_scat[0].data', stream_scat[0].data)

        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        
        
        trace = stream_scat[0]
        

        
        ax1 = plt.subplot(211)
        ax1.plot(trace.times("matplotlib"), stream_scat[0].data, color='black', lw=0.5)
        ax1.set_xlim(start_date, end_date)
        ax1.set_ylabel('strain', fontsize=12)
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5) 
        ax1.tick_params(axis='both', which='major', length=6, width=1)  
        ax1.tick_params(axis='both', which='minor', length=4, width=0.75)
        ax1.tick_params(which='both', direction='out')
        ### 複数日の場合は適切な間隔に設定
        if (end_date - start_date).days > 1:
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 3, 6, 12, 15, 18, 21]))
        else:
            ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24), interval=3))
            ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0,24), interval=1))

        
        
        ax2 = plt.subplot(212)
        ### plot cumulative number of samples in each cluster
        # sort by time to ensure monotonic cumulative plots
        sort_idx = np.argsort(times)
        times_sorted = times[sort_idx]
        preds_sorted = predictions[sort_idx]

        unique_predictions = np.unique(preds_sorted)
        max_cum = 0

        for cluster_id in unique_predictions:
            hits = (preds_sorted == cluster_id).astype(int)
            cum_counts = np.cumsum(hits)
            Nevents = cum_counts.max()
            max_cum = max(max_cum, int(cum_counts[-1]) if cum_counts.size > 0 else 0)
            if cluster_id<=7:
                ax2.plot(times_sorted, cum_counts / Nevents, label=f'Cluster {cluster_id} ({int(Nevents)})', linewidth=1.5)
            if cluster_id>7:
                ax2.plot(times_sorted, cum_counts / Nevents, label=f'Cluster {cluster_id} ({int(Nevents)})', linewidth=1.5, ls='--')

        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)
        ax2.set_ylabel('cumulative number (norm.)', fontsize=12)
        ax2.set_xlim(start_date, end_date)
        ax2.set_ylim(0, 1)
        
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5) 
        ax2.tick_params(axis='both', which='major', length=6, width=1)  
        ax2.tick_params(axis='both', which='minor', length=4, width=0.75)
        ax2.tick_params(which='both', direction='out')
        ### 複数日の場合は適切な間隔に設定
        if (end_date - start_date).days > 1:
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        else:
            ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24), interval=3))
            ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0,24), interval=1))
        

        plt.suptitle(f"{ustation} {start_date.strftime('%Y.%m.%d')}-{end_date.strftime('%Y.%m.%d')}", fontsize=12)

        output_filename_clustering = f"{output_dir}clustering_{ustation}_{date_range_str}.png"
        print(f"save clustering figure: {output_filename_clustering}")
        
        plt.savefig(output_filename_clustering, dpi=300, bbox_inches="tight")
        plt.close()
    
        # ---------------------------------------------
        # Plot example waveforms per cluster (up to 10)
        # ---------------------------------------------
        try:
            trace_full = stream_scat[0]
            # Use configured segment duration seconds from Params
            seg_dur = float(segment_duration_seconds)
            half = seg_dur / 2.0
            x_ref = None  # reuse x-axis if lengths match

            # Prepare cluster IDs sorted
            cluster_ids = np.unique(predictions)
            for cid in cluster_ids:
                # indices for this cluster
                hit_idx = np.where(predictions == cid)[0]
                if hit_idx.size == 0:
                    continue
                # choose up to 10 examples closest to the cluster center in feature space
                try:
                    cluster_feats = features[hit_idx]
                    center = np.nanmean(cluster_feats, axis=0)
                    # compute distances to centroid (robust to NaNs)
                    diff = np.nan_to_num(cluster_feats - center, copy=False)
                    dists = np.linalg.norm(diff, axis=1)
                    order = np.argsort(dists)
                    n_show = int(min(10, hit_idx.size))
                    chosen_idx = hit_idx[order[:n_show]]
                except Exception:
                    # fallback to earliest-in-time if any issue arises
                    hit_times = times[hit_idx]
                    order = np.argsort(hit_times)
                    hit_idx = hit_idx[order]
                    n_show = int(min(10, hit_idx.size))
                    chosen_idx = hit_idx[:n_show]

                ncols = 2
                nrows = int(np.ceil(n_show / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(10, 1.6 * nrows), sharex=False)
                axes = np.array(axes).reshape(nrows, ncols)

                for k, idx_k in enumerate(chosen_idx):
                    row = k // ncols
                    col = k % ncols
                    axk = axes[row, col]

                    t_center = times[idx_k]
                    t1 = obspy.UTCDateTime(t_center - datetime.timedelta(seconds=half))
                    t2 = obspy.UTCDateTime(t_center + datetime.timedelta(seconds=half))
                    seg = trace_full.slice(t1, t2)

                    y = seg.data.astype(float) if seg.data is not None else np.array([])
                    # optional: high-pass filter for visibility
                    try:
                        seg_f = seg.copy()
                        seg_f.detrend('linear')
                        seg_f.filter('highpass', freq=0.1, corners=2, zerophase=True)
                        y = seg_f.data.astype(float)
                    except Exception:
                        pass

                    if y.size == 0:
                        axk.text(0.5, 0.5, 'no data', ha='center', va='center')
                        axk.axis('off')
                        continue

                    # normalize
                    amp = np.nanmax(np.abs(y)) if np.any(np.isfinite(y)) else 1.0
                    if amp == 0:
                        amp = 1.0
                    y_norm = y / amp

                    # x axis in seconds relative to center
                    x = np.linspace(-half, half, y_norm.size)
                    print()
                    axk.plot(x, y_norm, lw=0.6, color=f'C{cid-1}')
                    axk.set_xlim(-half, half)
                    axk.set_ylim(-1.2, 1.2)
                    axk.set_title(t_center.strftime('%Y-%m-%d %H:%M:%S'), fontsize=9)
                    axk.grid(False)
                    for spine in axk.spines.values():
                        spine.set_linewidth(1.0)
                    axk.tick_params(axis='both', which='major', length=4, width=1)
                    axk.tick_params(axis='both', which='minor', length=2, width=0.75)

                # hide unused subplots
                total_axes = nrows * ncols
                for k in range(n_show, total_axes):
                    row = k // ncols
                    col = k % ncols
                    axes[row, col].axis('off')

                fig.suptitle(f"{ustation} cluster {cid} (N={np.sum(predictions==cid)})\nHigh-pass 0.1 Hz, normalized", fontsize=12)
                fig.supylabel('Normalized amplitude', fontsize=12)
                fig.supxlabel('time [s]', fontsize=12)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                out_waveforms = f"{output_dir}waveforms_cluster{cid}_{ustation}_{date_range_str}.png"
                print(f"save cluster example waveforms: {out_waveforms}")
                fig.savefig(out_waveforms, dpi=300, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"Waveform example plotting skipped due to error: {e}")

    print(f"\n=== Clustering completed for multi-day data ===")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Total samples: {len(predictions)}")
    print(f"Number of clusters: {Nclusters}")





