import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isfile, join
from scipy import stats
from scipy.optimize import curve_fit
import math
from sklearn.metrics import r2_score
from collections import defaultdict
from scipy.stats import binned_statistic

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 8}

plt.rc('font', **font)


"""
def func(x, a, b, c):
    return a * np.log(b * x) + c
"""

def func(x, a, b):
    return a * x + b 

mypath = str(sys.argv[1])
files = sorted([f for f in listdir(mypath) if isfile(join(mypath, f)) and ".txt" in f])
#files = [sys.argv[1], sys.argv[2], sys.argv[3]]

mypath2 = mypath + "/oversample/"
print mypath2
oversample_files = sorted([f for f in listdir(mypath2) if isfile(join(mypath2, f)) and ".txt" in f])
print oversample_files
#ctx_switch_lst, interrupts_lst, interrupts_adj_lst, stolen_ticks_lst, bench_lst, bill_lst


oversample_bench = []
oversample_stolen = []

for f in oversample_files:
    with open(mypath2 + str(f), 'r') as opened_f:
        val = opened_f.read()
        values = filter(lambda x: x != '' ,val.split("\n"))
        #print values

        for x in values:
            splt = x.split("\t")
            print splt
            oversample_bench.append(splt[1])
            oversample_stolen.append(splt[0])


ctxs_switch = []
int_lst = []
int_adj = []
stolen = []
bench = []
bench_cpu_tick = []
bill = []
first_stolen = []
first_bench = []
for f in files:
    ctxs = []

    zone_ctx = defaultdict(list)
    zone_times = defaultdict(list)

    stolen_per_probe = defaultdict(int)
    bench_per_probe = defaultdict(int)

    times = []

    stolen_subset = []
    bench_subset = []

    ef_bench = []
    ef_first_bench = []

    stolen_ratio_lst = []

    raw_perf = []

    with open(mypath + str(f), 'r') as opened_f:
        val = opened_f.read()
        values = filter(lambda x: x != '' ,val.split("\n"))
        print values

        for x in values:
            splt = x.split("\t")
            print splt
            ctxs_switch.append(float(splt[0]))
            int_lst.append(float(splt[1]))
            int_adj.append(float(splt[2]))
            stolen.append(float(splt[3]))
            if len(splt) > 10:
                bench.append(float(splt[10]))
            else:
                bench.append(float(splt[4]))
            bench_cpu_tick.append(float(splt[4]))

            bill.append(float(splt[5]))
            first_stolen.append(float(splt[7]))
            first_bench.append(float(splt[6]))
            stolen_ratio_lst.append(float(splt[9]))

            if len(splt) > 10:
                raw_perf.append(float(splt[10]))

            stolen_per_probe[splt[8]] += float(splt[3])
            bench_per_probe[splt[8]] += float(splt[4])

            if float(splt[3]) > 5:
                stolen_subset.append((float(splt[3])))
                bench_subset.append(float(splt[4]))

            # not first invocation on a function
            if len(splt) > 10:
                if float(splt[10]) != float(splt[11]):
                    ef_bench.append(float(splt[11]))
                    ef_first_bench.append(float(splt[10]))
            elif float(splt[4]) != float(splt[6]):
                ef_bench.append(float(splt[4]))
                ef_first_bench.append(float(splt[6]))

    font = {'family' : 'sans-serif',
            'weight' : 'bold',
            'size'   : 11}

    plt.rc('font', **font)
    

    probe_bench = []
    probe_stolen = []
    for (a, b), (c, d) in zip(bench_per_probe.items(), stolen_per_probe.items()):
        probe_bench.append(b)
        probe_stolen.append(d)

    probe_bench = np.array(probe_bench, dtype=float)
    probe_stolen = np.array(probe_stolen, dtype=float)
    plt.scatter(probe_stolen, probe_bench)
    """
    y_lim = plt.gca().get_ylim()
    x_lim = plt.gca().get_xlim()
    plt.xlim(x_lim[0], x_lim[1] *1.3)
    plt.ylim(y_lim[0], y_lim[1] *1.3)
    """
    plt.xlabel("Cumulative amount of stolen CPU ticks",
                fontname="Arial", fontsize=12, weight='bold')
    plt.ylabel("Total execution time of benchmark per function (User CPU Ticks)",
                fontname="Arial", fontsize=12, weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    y_lim = ax.get_ylim()

    ax.set_xlim(0, 80)
    ax.set_ylim(0, 10000)
    plt.savefig(mypath +"/perfn_stolen.png", bbox_inches='tight')
    plt.clf()

    ctxs_switch = np.array(ctxs_switch, dtype=float) 
    bench = np.array(bench, dtype=float)
    plt.scatter(ctxs_switch, bench)    
    plt.xlabel("Number of Context Switches",
                fontname="Arial", fontsize=12, weight='bold')
    plt.ylabel("Execution time of benchmark (User CPU Ticks)",
                fontname="Arial", fontsize=12, weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.savefig(mypath +"/contextswitch.png", bbox_inches='tight')
    plt.clf()

    int_lst = np.array(int_lst, dtype=float) 
    #bench = np.array(bench, dtype=float)
    plt.scatter(int_lst, bench)    
    plt.xlabel("Number of Interrupts Serviced",
                fontname="Arial", fontsize=12, weight='bold')
    plt.ylabel("Execution time of benchmark (User CPU Ticks)",
                fontname="Arial", fontsize=12, weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig(mypath +"/interrupts.png", bbox_inches='tight')
    plt.clf()

    int_adj = np.array(int_adj, dtype=float) 
    #bench = np.array(bench, dtype=float)
    plt.scatter(int_adj, bench)   
    plt.xlim(0, 1000)
    plt.xlabel("Number of Interrupts Serviced",
                fontname="Arial", fontsize=12, weight='bold')
    plt.ylabel("Execution time of benchmark (User CPU Ticks)",
                fontname="Arial", fontsize=12, weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig(mypath +"/interrupts_adj.png", bbox_inches='tight')
    plt.clf()

    stolen_ratio = np.array(stolen_ratio_lst, dtype=float)
    raw_perf = np.array(raw_perf, dtype=float) 
    fig = plt.figure()
    ax = plt.subplot(111)
    if len(stolen_ratio) == len(raw_perf):
        ax.scatter(stolen_ratio, raw_perf, label='Points from original dataset')
    #ax.scatter(over_stolen, over_bench, color='red', label='Oversampled points')
    #plt.xlim(0, max(stolen_ratio) * 1.3 )
    #plt.ylim(0, max(bench) * 1.3 )
    plt.xlabel("% of CPU Ticks Stolen",
                fontname="Arial", fontsize=12, weight='bold')
    plt.ylabel("Execution time of benchmark (ms)",
                fontname="Arial", fontsize=12, weight='bold')
    ax.legend()
    y_lim = plt.gca().get_ylim()
    x_lim = plt.gca().get_xlim()
    plt.xlim(x_lim[0], x_lim[1] *1.05)
    plt.ylim(y_lim[0], y_lim[1] *1.05)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(0, x_lim[1])
    plt.savefig(mypath +"/ratiostolen_raw_perf.png", bbox_inches='tight')
    plt.clf()

    over_stolen = np.array(oversample_stolen, dtype=float) 
    over_bench = np.array(oversample_bench, dtype=float)

    bench_cpu_tick = np.array(bench_cpu_tick, dtype=float)
    stolen_ratio = np.array(stolen_ratio_lst, dtype=float)
    #over_stolen_ratio = (over_stolen / (over_bench + over_stolen)) * 100

    # projected 

    print stolen
    print bench
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(stolen_ratio, bench_cpu_tick, label='Points from original dataset')
    ax.scatter(over_stolen, over_bench, color='red', label='Oversampled points')
    #plt.xlim(0, max(stolen_ratio) * 1.3 )
    #plt.ylim(0, max(bench) * 1.3 )
    plt.xlabel("% of CPU Ticks Stolen",
                fontname="Arial", fontsize=12, weight='bold')
    plt.ylabel("Execution time of benchmark (User CPU Ticks)",
                fontname="Arial", fontsize=12, weight='bold')
    ax.legend()
    y_lim = plt.gca().get_ylim()
    x_lim = plt.gca().get_xlim()
    plt.xlim(x_lim[0], x_lim[1] *1.05)
    plt.ylim(y_lim[0], y_lim[1] *1.05)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(0, x_lim[1])
    plt.savefig(mypath +"/ratiostolen.eps", bbox_inches='tight', format='eps')
    plt.clf()

    first_stolen = np.array(first_stolen, dtype=float) 
    #bench = np.array(bench, dtype=float)
    print stolen
    print bench
    plt.scatter(first_stolen, bench)
    """
    y_lim = plt.gca().get_ylim()
    x_lim = plt.gca().get_xlim()
    plt.xlim(x_lim[0], x_lim[1] *1.3)
    plt.ylim(y_lim[0], y_lim[1] *1.3)
    """
    plt.xlabel("Stolen CPU Ticks in first invocation of function")
    plt.ylabel("Execution time of benchmark (User CPU Ticks)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig(mypath +"/first_stolen.png", bbox_inches='tight')
    plt.clf()



    fig = plt.figure()
    ax = plt.subplot(111)
    ef_first_bench = np.array(ef_first_bench, dtype=float)
    ef_bench = np.array(ef_bench, dtype=float)
    #bench = np.array(bench, dtype=float)
    ax.scatter(ef_first_bench, ef_bench)
    #plt.xlim(min(first_bench), max(first_bench) * 1.5 )
    #plt.ylim(min(bench), max(bench) * 1.5 )
    print ef_first_bench, ef_bench
    z = np.polyfit(ef_first_bench, ef_bench, 1)
    p = np.poly1d(z)
    print p(ef_first_bench)
    r2 = r2_score(ef_bench, p(ef_first_bench))
    print ("y = " + str(z[0]) + "x + "+str(z[1]))
    ax.plot(ef_first_bench, p(ef_first_bench),"r--", label="r^2=" + str(r2))

    y_lim = plt.gca().get_ylim()
    x_lim = plt.gca().get_xlim()
    plt.xlim(x_lim[0], x_lim[1] *1.05)
    plt.ylim(y_lim[0], y_lim[1] *1.05)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel("Execution time of benchmark in the first function invocation (CPU Ticks)",
                fontname="Arial", fontsize=12, weight='bold')
    plt.ylabel("Execution time of benchmark (User CPU Ticks)",
                fontname="Arial", fontsize=12, weight='bold')
    ax.legend()
    plt.savefig(mypath +"/first_bench.png", bbox_inches='tight')
    plt.clf()


    std = np.std(bench)
    mean = np.mean(bench)

    print "std:\t", std
    print "mean:\t", mean
    print "CoV:\t", std / mean * 100
