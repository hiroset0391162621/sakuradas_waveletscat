import os
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
sys.path.append("scat/")
from network import ScatteringNetwork
from Params import *

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


if __name__ == "__main__":
    
    print(segment_duration_seconds, sampling_rate_hertz)
    samples_per_segment = int(segment_duration_seconds * sampling_rate_hertz)


    J_1st, Q_1st = 7, 4
    J_2nd, Q_2nd = 7, 2

    J_1st, Q_1st = 5, 4
    J_2nd, Q_2nd = 5, 2



    quality_1st, quality_2nd = 1, 3
    bank_keyword_arguments = (
        {"octaves": J_1st, "resolution": Q_1st, "quality": quality_1st},
        {"octaves": J_2nd, "resolution": Q_2nd, "quality": quality_2nd},
    )
    
    network = ScatteringNetwork(
        *bank_keyword_arguments,
        bins=samples_per_segment,
        sampling_rate=sampling_rate_hertz,
    )

    print(network.bins, network.sampling_rate)
    
    dirpath_save = "example/"

    # Create directory to save the results
    os.makedirs(dirpath_save, exist_ok=True)

    # Save the scattering network with Pickle
    filepath_save = os.path.join(dirpath_save, "scattering_network.pickle")
    with open(filepath_save, "wb") as file_save:
        pickle.dump(network, file_save, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    
    cmap = cm.get_cmap('plasma')


    # Loop over network layers
    layer_num = 1
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5,3))
    for bank in network.banks:

        # Create axes (left for temporal, right for spectral domain)
        
        

        # Show each wavelet
        fcenter_arr = list()
        i = 0
        for wavelet, spectrum, ratio in zip(
            bank.wavelets, bank.spectra, bank.ratios
        ):
            
            color = cmap(i / (len(bank)-1))  # カラーマップから色を取得
            # Spectral domain (log of amplitude)
            ax[layer_num-1].plot(bank.frequencies, np.log(np.abs(spectrum) + 1) + ratio, color=color)
            
            fcentre = bank.frequencies[np.argmax(np.log(np.abs(spectrum) + 1) + ratio)]
            fcenter_arr.append(fcentre)
            
            i += 1
        
        print('fc=', fcenter_arr[0], fcenter_arr[-1], len(fcenter_arr))

        # Limit view to three times the temporal width of largest wavelet
        width_max = 3 * bank.widths.max()
        
        # Labels
        ax[layer_num-1].set_xlim(0.01, 0.5*sampling_rate_hertz)
        ax[layer_num-1].set_xscale("log")
        
        ax[layer_num-1].grid()
        
        for spine in ax[layer_num-1].spines.values():
            spine.set_linewidth(1.5) 
        
        ax[layer_num-1].tick_params(axis='both', which='major', length=4, width=1)  
        ax[layer_num-1].tick_params(axis='both', which='minor', length=2, width=0.75)
        ax[layer_num-1].tick_params(which='both', direction='out')
    
        
        if layer_num==1:
            ax[layer_num-1].text(0.01, 2.5, '1st layer\n('+str(len(bank))+' wavelets)', fontsize=12, color='white', va='top')
        else:
            ax[layer_num-1].text(0.01, 2.5, '2nd layer\n('+str(len(bank))+' wavelets)', fontsize=12, color='white', va='top')
            
        ax[layer_num-1].set_facecolor('gray')
            
        layer_num += 1

    fig.supxlabel("Frequency [Hz]", fontsize=10, y=0.07)
    fig.supylabel("Octaves (base 2 log)", fontsize=10)
    plt.suptitle('filter banks', fontsize=12)
    plt.tight_layout()        
    #plt.savefig("filterbanks.pdf", dpi=100)
    #plt.savefig("filterbanks.png", dpi=100)
    plt.show()   
        
        
