import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isdir, join, isfile
from scipy import stats
from scipy.optimize import curve_fit
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from collections import defaultdict
from scipy.stats import binned_statistic
import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.cm as cm

def type1(plot):
    plot.rc('pdf',fonttype = 42)
    plot.rc('ps',fonttype = 42)
    """
    plot.rcParams['text.usetex'] = True #Let TeX do the typsetting
    plot.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
    plot.rcParams['font.family'] = 'sans-serif' # ... for regular text
    plot.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here
    """


def SetPlotRC():
    #If fonttype = 1 doesn't work with LaTeX, try fonttype 42.
    plt.rc('pdf',fonttype = 1)
    plt.rc('ps',fonttype = 1)

mypath = str(sys.argv[1])
subdirs = sorted([f for f in listdir(mypath) if isdir(join(mypath, f))])

"""
We want to collect per-run statistics
"""
r2_stolen = defaultdict(list)
r2_all = defaultdict(list)
cov_all = defaultdict(list)
std_all = defaultdict(list)
min_max = defaultdict(list)
sum_times = defaultdict(list)
week_averages = defaultdict(list)

stolen_ratio_all_99 = defaultdict(list)
stolen_ratio_all_75 = defaultdict(list)
stolen_ratio_all_50 = defaultdict(list)
stolen_ratio_all_25 = defaultdict(list)
stolen_ratio_combined = defaultdict(list)

perf_all_99 = defaultdict(list)
perf_all_75 = defaultdict(list)
perf_all_50 = defaultdict(list)
perf_all_25 = defaultdict(list)

perf_len_99 = defaultdict(list)
perf_len_75 = defaultdict(list)
perf_len_50 = defaultdict(list)
perf_len_25 = defaultdict(list)

variance_all = defaultdict(list)

net_perf_lst = []
net_jitter = []

net_avg_stolen = []
net_avg_perf = []

min_perf = defaultdict(list)
max_perf = defaultdict(list)

for directory in subdirs:
    sub_path = mypath + "/" + directory + "/"
    print sub_path
    benchmarks = sorted([f for f in listdir(sub_path) if isdir(join(sub_path, f))])
    print benchmarks

    for bench in benchmarks:
        benchmark_sub_path = sub_path + "/" + bench + "/"
        print benchmark_sub_path
        files = sorted([f for f in listdir(benchmark_sub_path) if isfile(join(benchmark_sub_path, f)) and ".txt" in f])
        benchmark_sub_path += files[0]
        with open(benchmark_sub_path, 'r') as opened_f:
            val = opened_f.read()
            values = filter(lambda x: x != '' ,val.split("\n"))

            bench_lst = []
            ef_bench = []
            ef_first_bench = []
            stolen_ratio_per_bench = []
            stolen_lst = []
            bill_lst = []
            jitter_temp = []
            outliers = []
            for x in values:
                splt = x.split("\t")
                bench_lst.append(float(splt[10]))
                stolen_ratio_per_bench.append(float(splt[9]))
                stolen_lst.append(float(splt[3]))
                bill_lst.append(float(splt[5]))
                if float(splt[4]) != float(splt[6]):
                    ef_bench.append(float(splt[4]))
                    ef_first_bench.append(float(splt[6]))
                if bench == 'net':
                    net_perf_lst.append(float(splt[5]))
                    jitter_temp.append((float(splt[10]), float(splt[12])))
                    if float(splt[10]) > 600:
                        outliers.append((float(splt[10]), float(splt[9])))
                    else:
                        jitter_temp.append((float(splt[10]), float(splt[12])))
                #if float(splt[9]) > 15:
                #    print splt

                """
                ctxs_switch.append(float(splt[0]))
                int_lst.append(float(splt[1]))
                int_adj.append(float(splt[2]))
                stolen.append(float(splt[3]))
                bench.append(float(splt[4]))
                bill.append(float(splt[5]))
                first_stolen.append(float(splt[7]))
                first_bench.append(float(splt[6]))
                stolen_ratio_lst.append(float(splt[9]))
                """

            bench_lst = np.array(bench_lst, dtype=float)

            min_perf[bench].append(np.min(bench_lst))
            max_perf[bench].append(np.max(bench_lst))

            sum_times[bench].extend(bench_lst)

            week_averages[bench].append(np.median(bench_lst))

            if bench == "net":
                temp = bench_lst
                bench_lst = filter(lambda l: l < 1000, bench_lst)
                # performance data per bench
                perf_all_99[bench].append(np.percentile(bench_lst, 99))
                perf_all_75[bench].append(np.percentile(bench_lst, 75))
                perf_all_50[bench].append(np.percentile(bench_lst, 50))
                perf_all_25[bench].append(np.percentile(bench_lst, 25))
                #min_perf[bench].append(np.min(bench_lst))
                #max_perf[bench].append(np.max(bench_lst))

                # compute cov
                std = np.std(bench_lst)
                mean = np.mean(bench_lst)
                #std = np.std(bill_lst)
                #mean = np.mean(bill_lst)
                cov_all[bench].append(std / mean * 100)
                bench_lst = temp
            else:
                perf_all_99[bench].append(np.percentile(bench_lst, 99))
                perf_all_75[bench].append(np.percentile(bench_lst, 75))
                perf_all_50[bench].append(np.percentile(bench_lst, 50))
                perf_all_25[bench].append(np.percentile(bench_lst, 25))
                #min_perf[bench].append(np.min(bench_lst))
                #max_perf[bench].append(np.max(bench_lst))
                std = np.std(bench_lst)
                mean = np.mean(bench_lst)
                #std = np.std(bill_lst)
                #mean = np.mean(bill_lst)
                cov_all[bench].append(std / mean * 100)
                std_all[bench].append(std)
            # number of functions
            cutoff1 = np.percentile(bench_lst, 25)
            cutoff2 = np.percentile(bench_lst, 50)
            cutoff3 = np.percentile(bench_lst, 75)
            cutoff4 = np.percentile(bench_lst, 99)
            perf_len_99[bench].append(len(filter(lambda l: cutoff4 <= l, bench_lst)))
            perf_len_75[bench].append(len(filter(lambda l: cutoff3 <= l <= cutoff4, bench_lst)))
            perf_len_50[bench].append(len(filter(lambda l: cutoff2 <= l <= cutoff3, bench_lst)))
            perf_len_25[bench].append(len(filter(lambda l: cutoff1 <= l <= cutoff2, bench_lst)))

            # stolen ratio per bench
            stolen_ratio_per_bench = np.array(stolen_ratio_per_bench, dtype=float)
            stolen_ratio_all_99[bench].append(np.percentile(stolen_ratio_per_bench, 99))
            stolen_ratio_all_75[bench].append(np.percentile(stolen_ratio_per_bench, 75))
            stolen_ratio_all_50[bench].append(np.percentile(stolen_ratio_per_bench, 50))
            stolen_ratio_all_25[bench].append(np.percentile(stolen_ratio_per_bench, 25))
            stolen_ratio_combined[bench].append(stolen_ratio_per_bench)

            # min vs max performance gap

            min_max[bench].append(((np.max(bench_lst) - np.min(bench_lst)) / np.min(bench_lst) * 100))

            # compute the line of best fit for perf stability/predictability
            ef_first_bench = np.array(ef_first_bench, dtype=float)
            ef_bench = np.array(ef_bench, dtype=float)
            z = np.polyfit(ef_bench, ef_first_bench, 1)
            print z
            p = np.poly1d(z)
            print p
            r2 = r2_score(ef_bench, p(ef_first_bench))
            mae = mean_absolute_error(ef_bench, p(ef_first_bench))
            mse = mean_squared_error(ef_bench, p(ef_first_bench))
            if r2 < 0:
                r2 = 0
            r2_all[bench].append(r2)

            # compute line of best for for stolen CPU versus bench w/log curve
            stolen_lst = np.array(stolen_ratio_per_bench, dtype=float)
            z = np.polyfit(stolen_lst, bench_lst, 1)
            p = np.poly1d(z)
            r2 = r2_score(bench_lst, p(stolen_lst))
            print ("stolen r2:\t" + str(r2))
            r2_stolen[bench].append(r2)

            # compute variance
            var = np.var(bench_lst)
            variance_all[bench].append(var)

            # compute jitter
            if bench == 'net':
                in_order_lst = [x for x, y in sorted(jitter_temp, key=lambda l: l[1])]
                perf = [x for x, y in outliers]
                stolen = [y for x, y in outliers]

                #filtered_benches = filter(lambda l: l, bench_lst)
                filtered_benches = filter(lambda l: l < 2000, bench_lst)
                #net_jitter.append(np.var(bench_lst) / np.average(stolen_lst))
                #net_jitter.append((np.max(bench_lst) - np.min(bench_lst)) / np.average(stolen_lst))
                filtered_net = filter(lambda l: l < 1000, in_order_lst)
                net_jitter.append(np.var(filtered_net))
                #net_jitter.append(np.var(filtered_benches))
                net_avg_perf.append(np.average(filtered_net))
                net_avg_stolen.append(np.average(stolen_ratio_per_bench))

def moving_average(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

# convienient functions for plotting
color_dict = defaultdict(str)
color_dict['cache'] = 'red'
color_dict['nqueens'] = 'green'
color_dict['video'] = 'blue'
color_dict['net'] = 'purple'

# marker
marker_dict = defaultdict(str)
marker_dict['cache'] = 'd'
marker_dict['video'] = 'o'
marker_dict['nqueens'] = 's'
marker_dict['net'] = '*'

# marker size
marker_size_dict = defaultdict(int)
marker_size_dict['cache'] = 6
marker_size_dict['video'] = 6
marker_size_dict['nqueens'] = 6
marker_size_dict['net'] = 6


for x, y in marker_dict.items():
    print ("Daily min-max for:\t" + str(x))
    try:
        print ("max:\t" + str(np.max(max_perf[x])))
        print ("min:\t" + str(np.min(min_perf[x])))
        print ("std:\t" + str(np.average(cov_all[x])))
        print ((np.max(max_perf[x]) - np.min(min_perf[x])) / np.min(min_perf[x]) * 100)
    except:
        print (str(x) + ": not present in dataset")





#start = datetime.datetime(2020, 1, 20, 00, 05)
#start  = datetime.datetime(2020, 3, 17, 00, 00)
start  = datetime.datetime(2020, 3, 31, 00, 00)
times = [start + datetime.timedelta(hours=i) for i in range(0,10000*7,2)]
#times = range(0,48,2)
times = times[::1]

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 12}

plt.rc('font', **font)

# plot net_jitter
fig = plt.figure()
ax = plt.subplot(111)

#jitter_start = datetime.datetime(2020, 1, 31, 23, 00)
#jitter_start = datetime.datetime(2020, 3, 17, 00, 00)
jitter_start = datetime.datetime(2020, 3, 31, 00, 00)
jitter_times = [jitter_start + datetime.timedelta(hours=i) for i in range(0,10000*7,2)]
jitter_labels = [jitter_start + datetime.timedelta(hours=i) for i in range(0,26*7,2)]
"""
print net_jitter
ax.plot(jitter_times[:len(net_jitter)], net_jitter, color='blue', linewidth=2)
ma = moving_average(net_jitter, 4)
ax.grid()

ax.plot(jitter_times[:len(ma)], ma, label="Moving Average (8 Hour Buckets)", color='black', linewidth=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_yscale('log')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.legend(prop={'size': 10})
y_lim = plt.gca().get_ylim()
x_lim = plt.gca().get_xlim()
#plt.xlim(x_lim[0], x_lim[1] *1.05)
#ax.set_xticks(jitter_labels[:len(ma)][::2])
#ax.set_xticklabels(jitter_labels[:len(ma)][::2], fontsize = 8, va='top', ha='left')
#ax.set_yticks(range(0,50,2))
#ax.set_yticklabels(range(0,50,2), fontsize = 12)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

plt.ylim(0, y_lim[1] *1.55)
#plt.yticks(np.arange(y_lim[0], y_lim[1] *1.35, step=2))
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xlabel("Time (ms)", fontname="Arial", fontsize=12, weight='bold')
plt.ylabel("Variance of End-to-end Execution Time", fontname="Arial", fontsize=12, weight='bold')
plt.savefig("net_jitter.eps", bbox_inches='tight', format='eps')
plt.clf()




print len(net_avg_stolen)
print len(net_jitter)
fig = plt.figure()
ax = plt.subplot(111)
dense = (0, (5, 1))
ax.grid()
ax.plot(jitter_times[:len(net_avg_perf)], net_avg_perf, color='blue', linewidth=2, dashes=[10, 5, 20, 5])

ma = moving_average(net_avg_perf, 4)

ax.plot(jitter_times[:len(ma)], ma, label="Moving Average (8 Hour Buckets)", color='black', linewidth=4)

ax.legend(prop={'size': 10})
y_lim = plt.gca().get_ylim()
x_lim = plt.gca().get_xlim()
plt.ylim(500, 520)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
plt.gca().xaxis.set_major_formatter(myFmt)

#plt.xlabel("Time (ms)", fontname="Arial", fontsize=12, weight='bold')
plt.ylabel("Average End-to-end Execution Time (ms)", fontname="Arial", fontsize=12, weight='bold')

plt.savefig("jitter_stolen.eps", bbox_inches='tight', format='eps')
plt.clf()



# plot cov
fig = plt.figure()
ax = plt.subplot(111)
l = 0

for key, lst in cov_all.items():
    l = len(lst)
    print "CoV:\t" + str(key), np.average(lst)
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], linewidth=2)
    ax.legend(prop={'size': 12})

y_lim = plt.gca().get_ylim()
x_lim = plt.gca().get_xlim()
#plt.xlim(x_lim[0], x_lim[1] *1.05)
ax.grid()
ax.set_xticks(times[:len(lst)][::12])
ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')
ax.set_yticks(range(0,70,1))
ax.set_yticklabels(range(0,70,1), fontsize = 8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax.set_yscale('log')
plt.ylim(0, y_lim[1] *50)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_yscale('log')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#plt.yticks(np.arange(y_lim[0], y_lim[1] *1.35, step=2))
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xlabel("Time", fontname="Arial", fontsize=18, weight='bold')
plt.ylabel("Coefficient of Variation", fontname="Arial", fontsize=18, weight='bold')
plt.savefig("total_cov.eps", bbox_inches='tight')
plt.clf()


# plot max-min gap
fig = plt.figure()
ax = plt.subplot(111)
l = 0
for key, lst in min_max.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], linewidth=4)
    ax.legend(prop={'size': 12})
y_lim = plt.gca().get_ylim()
x_lim = plt.gca().get_xlim()
#plt.xlim(x_lim[0], x_lim[1] *1.05)
ax.set_xticks(times[:len(lst)][::12])
ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')
ax.set_yscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_yscale('log')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_yticks(range(0,50,2))
#ax.set_yticklabels(range(0,50,2), fontsize = 12)
#ax.yaxis.set_major_locator(ticker.MultipleLocator(4))

#plt.ylim(y_lim[0], y_lim[1] *1.55)
#plt.yticks(np.arange(y_lim[0], y_lim[1] *1.35, step=2))
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xlabel("Time", fontname="Arial", fontsize=18, weight='bold')
plt.ylabel("% Speedup of Min to Max", fontname="Arial", fontsize=18, weight='bold')
plt.savefig("maxmin_perf.eps", bbox_inches='tight')
plt.clf()
"""

"""
# plot r2
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
for ax in axes.flat:
    ax.set(xlabel='Time', ylabel='$R^2$')
ax = axes[0]
ax.set_title('Linear model (Performance Stability)')
for key, lst in r2_all.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], marker=marker_dict[key], markersize=marker_size_dict[key], linewidth=4)
    ax.legend(prop={'size': 12})
y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_xticks(times[:len(lst)][::12])
ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#plt.xlim(x_lim[0], x_lim[1] *1.05)
ax.set_ylim(y_lim[0], y_lim[1] *1.35)
#ax.set_yticklabels(np.array(range(0, 11, 1), dtype=float) / 10, fontsize = 12, va='top', ha='left')
myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)


# plot r2 stolen
ax = axes[1]
ax.set_title('Logarithmic model (% CPU Time Stolen vs. Execution Time (ms))')
for key, lst in r2_stolen.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], marker=marker_dict[key], markersize=marker_size_dict[key], linewidth=4)
    ax.legend(prop={'size': 12})
y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::12])
ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')

#plt.xlim(x_lim[0], x_lim[1] *1.05)
ax.set_ylim(y_lim[0], y_lim[1] *1.35)
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.savefig("r2_models.eps", bbox_inches='tight')
plt.clf()
"""

# plot stolen
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(30 * 0.75, 8))
type1(plt)
for ax in axes.flat:
    ax.set_xlabel('Time', fontsize=18, weight='bold')
    ax.set_ylabel('% Stolen CPU Time', fontsize=18, weight='bold')
    ax.grid()

ax = axes[0]
ax.set_title('99$^{th}$ Percentile')
for key, lst in stolen_ratio_all_99.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key],
            linewidth=4, marker=marker_dict[key], markersize=marker_size_dict[key])
    ax.legend(prop={'size': 12})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(0, y_lim[1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::12])
ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')

#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.55)
#ax.autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)

ax = axes[1]
ax.set_title('75$^{th}$ Percentile')
for key, lst in stolen_ratio_all_75.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], linewidth=4,
            marker=marker_dict[key], markersize=marker_size_dict[key])
    ax.legend(prop={'size': 12})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
#ax.set_ylim(0, 19000)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::12])
ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')

ax.set_ylim(0, y_lim[1])
#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.55)
#ax.autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)

ax = axes[2]
ax.set_title('50$^{th}$ Percentile')
for key, lst in stolen_ratio_all_50.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], linewidth=4,
            marker=marker_dict[key], markersize=marker_size_dict[key])
    ax.legend(prop={'size': 12})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(0, y_lim[1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::12])
ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')

#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.35)
#ax.autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)

ax = axes[3]
ax.set_title('25$^{th}$ Percentile')
for key, lst in stolen_ratio_all_25.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], linewidth=4,
            marker=marker_dict[key], markersize=marker_size_dict[key])
    ax.legend(prop={'size': 12})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(0, y_lim[1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::12])
ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')

plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)
fig.tight_layout()
plt.savefig("total_stolen_ticks.pdf", bbox_inches='tight')
plt.clf()


# plot raw performance
type1(plt)
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(30 * 0.75, 8))
for ax in axes.flat:
    ax.set_xlabel('Time', fontname="Arial", fontsize=18, weight='bold')
    ax.set_ylabel('Normalized Execution Time', fontname="Arial", fontsize=18, weight='bold')

ax = axes[0]
ax.set_title('cache')
ax.grid()

print "times", times[:len(perf_all_99['cache'])]

ax.xaxis_date()

avg_time = np.sum(sum_times['cache']) / len(sum_times['cache'])

ax.plot(times[:len(perf_all_99['cache'])], perf_all_99['cache'] / avg_time, label=" P$_{99}$",
        color='red', linewidth=4, marker='d', markersize=6)
ax.plot(times[:len(perf_all_99['cache'])], perf_all_75['cache'] / avg_time, label=" P$_{75}$",
        color='green', linewidth=4, marker='o', markersize=6)
ax.plot(times[:len(perf_all_99['cache'])], perf_all_50['cache'] / avg_time, label=" P$_{50}$",
        color='blue', linewidth=4, marker='s', markersize=6)
ax.plot(times[:len(perf_all_99['cache'])], perf_all_25['cache'] / avg_time, label=" P$_{25}$",
        color='purple', linewidth=4, marker='*', markersize=6)
ax.legend(prop={'size': 12})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()

ax.set_ylim(0.8, 1.86)
yticks = np.arange(0.8, 1.86, 0.05)
yticks_labels = ["{:0.1f}".format(tick) for tick in yticks]
yticks_temp = []
for x in range(len(yticks_labels)):
    if x % 2 == 0:
        yticks_temp.append(yticks_labels[x])
    else:
        yticks_temp.append("")
yticks_labels = yticks_temp
ax.set_yticks(yticks)
ax.set_yticklabels(yticks_labels)

#ax.set_yscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.set_xticks(times[:len(perf_all_99['cache'])][::12])
ax.set_xticklabels(times[:len(perf_all_99['cache'])][::12], fontsize = 12, va='top', ha='left')
#ax.set_xticks(times[:len(lst)][::12])
#ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')
#spacing = 1
#for label in ax.xaxis.get_ticklabels()[::spacing]:
#    label.set_visible(False)

#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.35)
plt.gcf().autofmt_xdate()

myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)


ax = axes[1]
ax.set_title('video')
ax.xaxis_date()
ax.grid()

avg_time = np.sum(sum_times['video']) / len(sum_times['video'])

ax.plot(times[:len(perf_all_99['video'])], perf_all_99['video'] / avg_time, label=" P$_{99}$", color='red', linewidth=4, marker='d', markersize=6)
ax.plot(times[:len(perf_all_99['video'])], perf_all_75['video'] / avg_time, label=" P$_{75}$", color='green', linewidth=4, marker='o', markersize=6)
ax.plot(times[:len(perf_all_99['video'])], perf_all_50['video'] / avg_time, label=" P$_{50}$", color='blue', linewidth=4, marker='s', markersize=6)
ax.plot(times[:len(perf_all_99['video'])], perf_all_25['video'] / avg_time, label=" P$_{25}$", color='purple', linewidth=4, marker='*', markersize=6)
ax.legend(prop={'size': 12})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(0.8, 1.86)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks_labels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(times[:len(lst)][::12])
#ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')
ax.set_xticks(times[:len(perf_all_99['cache'])][::12])
ax.set_xticklabels(times[:len(perf_all_99['cache'])][::12], fontsize = 12, va='top', ha='left')
#
#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.35)
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)

ax = axes[2]
ax.set_title('nqueens')
ax.xaxis_date()
ax.grid()

avg_time = np.sum(sum_times['nqueens']) / len(sum_times['nqueens'])

ax.plot(times[:len(perf_all_99['nqueens'])], perf_all_99['nqueens'] / avg_time, label=" P$_{99}$", color='red', linewidth=4, marker='d', markersize=6)
ax.plot(times[:len(perf_all_99['nqueens'])], perf_all_75['nqueens'] / avg_time, label=" P$_{75}$", color='green', linewidth=4, marker='o', markersize=6)
ax.plot(times[:len(perf_all_99['nqueens'])], perf_all_50['nqueens'] / avg_time, label=" P$_{50}$", color='blue', linewidth=4, marker='s', markersize=6)
ax.plot(times[:len(perf_all_99['nqueens'])], perf_all_25['nqueens'] / avg_time, label=" P$_{25}$", color='purple', linewidth=4, marker='*', markersize=6)
ax.legend(prop={'size': 12})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(0.8, 1.86)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(times[:len(lst)][::12])
#ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')
ax.set_xticks(times[:len(perf_all_99['cache']) + 1:2])
ax.set_xticklabels(times[:len(perf_all_99['cache']) + 1:2], fontsize = 12, va='top', ha='left')
ax.set_yticks(yticks)
ax.set_yticklabels(yticks_labels)
#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.35)
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)
locator = mdates.DayLocator()
ax.xaxis.set_major_locator(locator)

ax = axes[3]
ax.set_title('net')
ax.xaxis_date()
ax.grid()

avg_time = np.sum(sum_times['net']) / len(sum_times['net'])

ax.plot(times[:len(perf_all_99['net'])], perf_all_99['net'] / avg_time,
        label=" P$_{99}$", color='red', linewidth=4, marker='d', markersize=6)
ax.plot(times[:len(perf_all_99['net'])], perf_all_75['net'] / avg_time,
        label=" P$_{75}$", color='green', linewidth=4, marker='o', markersize=6)
ax.plot(times[:len(perf_all_99['net'])], perf_all_50['net'] / avg_time,
        label=" P$_{50}$", color='blue', linewidth=4, marker='s', markersize=6)
ax.plot(times[:len(perf_all_99['net'])], perf_all_25['net'] / avg_time,
        label=" P$_{25}$", color='purple', linewidth=4, marker='*', markersize=6)
ax.legend(prop={'size': 12}, loc="upper right")

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
print y_lim
print x_lim
ax.set_ylim(0.8, 1.86)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks_labels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(times[:len(lst)][::12])
#ax.set_xticklabels(times[:len(lst)][::12], fontsize = 12, va='top', ha='left')
ax.set_xticks(times[:len(perf_all_99['cache'])][::12])
ax.set_xticklabels(times[:len(perf_all_99['cache'])][::12], fontsize = 12, va='top', ha='left')
#
#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.35)
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)
fig.tight_layout()
fig.savefig("total_performance.pdf", bbox_inches='tight')
plt.clf()



# plot performance averages
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
for ax in axes.flat:
    ax.set_xlabel('Time', fontname="Arial", fontsize=18, weight='bold')
    ax.set_ylabel('Normalized Execution Time', fontname="Arial", fontsize=18, weight='bold')

ax = axes[0]
ax.set_title('nqueens')
ax.grid()


ax.xaxis_date()

avg_time = np.sum(sum_times['nqueens']) / len(sum_times['nqueens'])

ax.plot(times[:len(week_averages['nqueens'])], week_averages['nqueens'], label="Average",
        color='red', linewidth=2, marker='d', markersize=6)

ma = moving_average(week_averages['nqueens'], 8)
ax.plot(jitter_times[:len(ma)][4:-4], ma[4:-4], label="Moving Average (8 Hour Buckets)", color='black', linewidth=4)

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()

#ax.set_ylim(16000, 19000)
yticks = np.arange(0, 2.1, 0.05)
yticks_labels = ["{:0.1f}".format(tick) for tick in yticks]
yticks_temp = []
for x in range(len(yticks_labels)):
    if x % 2 == 0:
        yticks_temp.append(yticks_labels[x])
    else:
        yticks_temp.append("")
yticks_labels = yticks_temp
#ax.set_yticks(yticks)
#ax.set_yticklabels(yticks_labels)

#ax.set_yscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.set_xticks(times[:len(week_averages['nqueens'])][::12])
ax.set_xticklabels(times[:len(week_averages['nqueens'])][::12], fontsize = 12, va='top', ha='left')

plt.gcf().autofmt_xdate()

myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)


ax = axes[1]
ax.set_title('net')
ax.grid()


ax.xaxis_date()

avg_time = np.sum(sum_times['net']) / len(sum_times['net'])

ax.plot(times[:len(net_jitter)], net_jitter / np.average(net_jitter), label="Average",
        color='red', linewidth=2, marker='d', markersize=6)

ma = moving_average(net_jitter / np.average(net_jitter), 4)
ax.plot(jitter_times[:len(ma)][2:-1], ma[2:-1], label="Moving Average (8 Hour Buckets)", color='black', linewidth=4)

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()

#ax.set_ylim(0.9, 1.1)
yticks = np.arange(0, 2.1, 0.05)
yticks_labels = ["{:0.1f}".format(tick) for tick in yticks]
yticks_temp = []
for x in range(len(yticks_labels)):
    if x % 2 == 0:
        yticks_temp.append(yticks_labels[x])
    else:
        yticks_temp.append("")
yticks_labels = yticks_temp
#ax.set_yticks(yticks)
#ax.set_yticklabels(yticks_labels)

#ax.set_yscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.set_xticks(times[:len(net_jitter)][::12])
ax.set_xticklabels(times[:len(net_jitter)][::12], fontsize = 12, va='top', ha='left')

plt.gcf().autofmt_xdate()

myFmt = mdates.DateFormatter('%b, %d %a')
ax.xaxis.set_major_formatter(myFmt)

fig.tight_layout()
fig.savefig("averages.eps", bbox_inches='tight')
plt.clf()


