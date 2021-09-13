import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.animation as animation
import numpy as np
import json 


def pof2gnuplot(history_files, fout="pof.dat", keys = None):
    if not history["obj"] : return 
    
    nelems = len(history["obj"])
    if not keys : keys = history["obj"][0].keys

    with open(fout, 'w') as file:
        file.write("#")
        for k in keys : file.write("{} ".format(k))
        file.write('\n')
        for i in range(nelems):
            if history["fn"][i] != '1.0': continue
            for k in keys : file.write("{} ".format(history["obj"][i][k]))
            file.write('\n')



def load_pof(fname, keys = None):
    with open(fname) as file:
        history = json.load(file)

    if not history["obj"] : return 
    
    nelems = len(history["obj"])
    if not keys : keys = history["obj"][0].keys()

    key_tuples = [eval(k) for k in keys]

    pof = [[history["obj"][i][k]*t[1] for k,t in zip(keys, key_tuples)] for i in range(nelems) if  history["fn"][i] == '1.0']

    return pof, tuple(key_tuples)

def expand_ranges(min, max, s=0.05):
    d = s*(max - min)
    return min - d, max + d 

def plt_pof(history_files, scales = None):
    pofs = []
    for f in history_files:
        pof, keys = load_pof(f)
        pofs.append(pof)

    if scales : 
        sf = [ s[1] for s in scales]
    else:
        sf = [ 1.0 for _ in keys]

    nppof = [[] for _ in range(len(keys) -1)] 
    for pof in range(len(pofs)):
        xy = np.array(pofs[pof])
        c = np.zeros(xy.shape[0]) + 0.2*pof
        s = np.zeros(xy.shape[0]) + 5
        for p in range(len(keys) -1):
            
            nppof[p].append(np.c_[xy[:,0]*sf[0], xy[:,p+1]*sf[p+1], c, s])

    nppof = [ np.vstack(n) for n in nppof]
            
    fig, ax = plt.subplots(len(nppof), sharex=True)
    fig.set_dpi(300)

    for p in range(len(nppof)):
        scat = ax[p].scatter(nppof[p][:,0], nppof[p][:,1], c=nppof[p][:,2], s=nppof[p][:,3])
        xmin, xmax = expand_ranges(nppof[p][:,0].min(), nppof[p][:,0].max())
        ymin, ymax = expand_ranges(nppof[p][:,1].min(), nppof[p][:,1].max())

        ax[p].axis([xmin, xmax, ymin, ymax])
        ax[p].set_ylabel(keys[p+1][0] + (scales[p+1][0] if scales else ""))
    
    ax[len(nppof) - 1].set_xlabel(keys[0][0]+ (scales[0][0] if scales else ""))
    plt.show()
