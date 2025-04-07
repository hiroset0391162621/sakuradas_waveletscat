import numpy as np

def cosTaper(windL, percent):
    N = windL
    tp = np.ones(N)
    for i in range(int(N*percent+1)):
        tp[i] *= 0.5 * (1 - np.cos((np.pi * i) / ( N * percent)))

    for i in range(int(N*(1-percent)), N):
        tp[i] *= 0.5 * (1 - np.cos((np.pi * (i+1)) / ( N * percent)))

    return tp