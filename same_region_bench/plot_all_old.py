import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isdir, join, isfile
from scipy import stats
from scipy.optimize import curve_fit
import math
from sklearn.metrics import r2_score
from collections import defaultdict
from scipy.stats import binned_statistic
import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.cm as cm


mypath = str(sys.argv[1])
subdirs = sorted([f for f in listdir(mypath) if isdir(join(mypath, f))])

"""
We want to collect per-run statistics
"""
r2_stolen = defaultdict(list)
r2_all = defaultdict(list)
cov_all = defaultdict(list)
min_max = defaultdict(list)

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
            for x in values:
                splt = x.split("\t")
                bench_lst.append(float(splt[4]))
                stolen_ratio_per_bench.append(float(splt[9]))
                stolen_lst.append(float(splt[3]))
                bill_lst.append(float(splt[5]))
                if float(splt[4]) != float(splt[6]):
                    ef_bench.append(float(splt[4]))
                    ef_first_bench.append(float(splt[6]))
                if bench == 'net':
                    net_perf_lst.append(float(splt[5]))
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

            # performance data per bench
            bench_lst = np.array(bench_lst, dtype=float)
            perf_all_99[bench].append(np.percentile(bench_lst, 99))
            perf_all_75[bench].append(np.percentile(bench_lst, 75))
            perf_all_50[bench].append(np.percentile(bench_lst, 50))
            perf_all_25[bench].append(np.percentile(bench_lst, 25))

            # number of functions
            cutoff1 = np.percentile(bill_lst, 25)
            cutoff2 = np.percentile(bill_lst, 50)
            cutoff3 = np.percentile(bill_lst, 75)
            cutoff4 = np.percentile(bill_lst, 99)
            perf_len_99[bench].append(len(filter(lambda l: cutoff4 <= l, bill_lst)))
            perf_len_75[bench].append(len(filter(lambda l: cutoff3 <= l <= cutoff4, bill_lst)))
            perf_len_50[bench].append(len(filter(lambda l: cutoff2 <= l <= cutoff3, bill_lst)))
            perf_len_25[bench].append(len(filter(lambda l: cutoff1 <= l <= cutoff2, bill_lst)))

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
            z = np.polyfit(ef_first_bench, ef_bench, 1)
            p = np.poly1d(z)
            r2 = r2_score(ef_bench, p(ef_first_bench))
            r2_all[bench].append(r2)

            # compute line of best for for stolen CPU versus bench w/log curve
            stolen_lst = np.array(stolen_ratio_per_bench, dtype=float)
            z = np.polyfit(stolen_lst, bench_lst, 1)
            p = np.poly1d(z)
            r2 = r2_score(bench_lst, p(stolen_lst))
            print ("stolen r2:\t" + str(r2))
            r2_stolen[bench].append(r2)

            # compute cov
            std = np.std(bench_lst)
            mean = np.mean(bench_lst)
            #std = np.std(bill_lst)
            #mean = np.mean(bill_lst)
            cov_all[bench].append(std / mean * 100)

            # compute variance
            var = np.var(bill_lst)
            variance_all[bench].append(var)

            # compute jitter
            #if bench == 'net':
                #in_order_lst = [x for x, y in sorted(jitter_temp, key=lambda l: l[1])]
            """
                temp = []
                for sample in xrange(len(in_order_lst) - 1):
                    temp.append(abs(in_order_lst[sample] - in_order_lst[sample + 1]))
                net_jitter.append(sum(temp) / len(temp))
            """
                #net_jitter.append(np.var(in_order_lst))

print net_jitter

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
marker_size_dict['video'] = 4
marker_size_dict['nqueens'] = 4
marker_size_dict['net'] = 8

start = datetime.datetime(2020, 1, 20, 02, 05)
times = [start + datetime.timedelta(hours=i) for i in range(0,48,2)]
#times = range(0,48,2)

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 8}

plt.rc('font', **font)

# plot net jitter
fig = plt.figure()
ax = plt.subplot(111)
l = 0
ax.plot(times[:len(net_jitter)], net_jitter, label=str('net'), color=color_dict['net'], linewidth=4)
ax.legend(prop={'size': 10})
y_lim = plt.gca().get_ylim()
x_lim = plt.gca().get_xlim()
#plt.xlim(x_lim[0], x_lim[1] *1.05)
ax.set_xticks(times[:len(net_jitter)])
ax.set_xticklabels(times[:len(net_jitter)], fontsize = 8, va='top', ha='left')
#ax.set_yticks(range(0,50,2))
#ax.set_yticklabels(range(0,50,2), fontsize = 12)
#ax.yaxis.set_major_locator(ticker.MultipleLocator(4))

#plt.ylim(y_lim[0], y_lim[1] *1.55)
#plt.yticks(np.arange(y_lim[0], y_lim[1] *1.35, step=2))
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xlabel("Time", fontname="Arial", fontsize=8, weight='bold')
plt.ylabel("Jitter", fontname="Arial", fontsize=8, weight='bold')
plt.savefig("net_jitter.png", bbox_inches='tight')
plt.clf()



# plot cov
fig = plt.figure()
ax = plt.subplot(111)
l = 0
for key, lst in cov_all.items():
    l = len(lst)
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], linewidth=2, marker=marker_dict[key], markersize=marker_size_dict[key])
    ax.legend(prop={'size': 10})
y_lim = plt.gca().get_ylim()
x_lim = plt.gca().get_xlim()
#plt.xlim(x_lim[0], x_lim[1] *1.05)
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 8, va='top', ha='left')
ax.set_yticks(range(0,50,2))
ax.set_yticklabels(range(0,50,2), fontsize = 12)
ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.ylim(y_lim[0], y_lim[1] *1.55)
plt.yticks(np.arange(y_lim[0], y_lim[1] *1.35, step=2))
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xlabel("Time", fontname="Arial", fontsize=12, weight='bold')
plt.ylabel("Coefficient of variation", fontname="Arial", fontsize=12, weight='bold')
plt.savefig("total_cov.eps", bbox_inches='tight', format='eps')
plt.clf()


# plot max-min gap
fig = plt.figure()
ax = plt.subplot(111)
l = 0
for key, lst in min_max.items():
    l = len(lst)
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], linewidth=4)
    ax.legend(prop={'size': 12})
y_lim = plt.gca().get_ylim()
x_lim = plt.gca().get_xlim()
#plt.xlim(x_lim[0], x_lim[1] *1.05)
ax.set_xticks(times[:len(lst)])
ax.set_xticklabels(times[:len(lst)], fontsize = 12, va='top', ha='left')
ax.set_yscale('log')
#ax.set_yticks(range(0,50,2))
#ax.set_yticklabels(range(0,50,2), fontsize = 12)
#ax.yaxis.set_major_locator(ticker.MultipleLocator(4))

#plt.ylim(y_lim[0], y_lim[1] *1.55)
#plt.yticks(np.arange(y_lim[0], y_lim[1] *1.35, step=2))
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xlabel("Time", fontname="Arial", fontsize=11, weight='bold')
plt.ylabel("% Speedup of Min to Max", fontname="Arial", fontsize=11, weight='bold')
plt.savefig("maxmin_perf.png", bbox_inches='tight')
plt.clf()



# plot r2
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))
for ax in axes.flat:
    ax.set(xlabel='Time', ylabel='r^2')
ax = axes[0]
ax.set_title('Linear model (Performance Stability)')
for key, lst in r2_all.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], marker=marker_dict[key], markersize=marker_size_dict[key], linewidth=2)
    ax.legend(prop={'size': 8})
y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 8, va='top', ha='left')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#plt.xlim(x_lim[0], x_lim[1] *1.05)
ax.set_ylim(y_lim[0], y_lim[1] *1.35)
myFmt = mdates.DateFormatter('%b, %d %H:%M')
ax.xaxis.set_major_formatter(myFmt)


# plot r2 stolen
ax = axes[1]
ax.set_title('Logarithmic model (Stolen Ticks vs. User Ticks)')
for key, lst in r2_stolen.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], marker=marker_dict[key], markersize=marker_size_dict[key], linewidth=2)
    ax.legend(prop={'size': 8})
y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 8, va='top', ha='left')

#plt.xlim(x_lim[0], x_lim[1] *1.05)
ax.set_ylim(y_lim[0], y_lim[1] *1.35)
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.savefig("r2_models.eps", bbox_inches='tight', format='eps')
plt.clf()


# plot stolen
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))
for ax in axes.flat:
    ax.set_xlabel('Time', fontname="Arial", fontsize=8, weight='bold')
    ax.set_ylabel('% Stolen CPU Ticks', fontname="Arial", fontsize=8, weight='bold')

ax = axes[0, 0]
ax.set_title('99th Percentile')
for key, lst in stolen_ratio_all_99.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key],
            linewidth=2, marker=marker_dict[key], markersize=marker_size_dict[key])
    ax.legend(prop={'size': 10})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(y_lim[0], y_lim[1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 8, va='top', ha='left')

#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.55)
#ax.autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
ax.xaxis.set_major_formatter(myFmt)

ax = axes[0, 1]
ax.set_title('75th Percentile')
for key, lst in stolen_ratio_all_75.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], linewidth=2,
            marker=marker_dict[key], markersize=marker_size_dict[key])
    ax.legend(prop={'size': 10})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(y_lim[0], y_lim[1] *1.35)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 8, va='top', ha='left')

#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.55)
#ax.autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
ax.xaxis.set_major_formatter(myFmt)

ax = axes[1, 0]
ax.set_title('50th Percentile')
for key, lst in stolen_ratio_all_50.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], linewidth=2,
            marker=marker_dict[key], markersize=marker_size_dict[key])
    ax.legend(prop={'size': 10})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(y_lim[0], y_lim[1] * 1.1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 8, va='top', ha='left')

#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.35)
#ax.autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
ax.xaxis.set_major_formatter(myFmt)

ax = axes[1, 1]
ax.set_title('25th Percentile')
for key, lst in stolen_ratio_all_25.items():
    ax.plot(times[:len(lst)], lst, label=str(key), color=color_dict[key], linewidth=2,
            marker=marker_dict[key], markersize=marker_size_dict[key])
    ax.legend(prop={'size': 10})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(y_lim[0], y_lim[1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 8, va='top', ha='left')

plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
ax.xaxis.set_major_formatter(myFmt)
fig.tight_layout()
plt.savefig("total_stolen_ticks.eps", bbox_inches='tight', format='eps')
plt.clf()


# plot raw performance
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))
"""
for ax in axes.flat:
    ax.set_xlabel('Time', fontname="Arial", fontsize=11, weight='bold')
    ax.set_ylabel('Benchmark Execution Time (Non-Nice User CPU Ticks)', fontname="Arial", fontsize=11, weight='bold')
"""

ax = axes[0, 0]
ax.set_title('cache')
ax.xaxis_date()
ax.plot(times[:len(perf_all_99['cache'])], perf_all_99['cache'], label=" 99%",
        color='red', linewidth=2, marker='d', markersize=6)
ax.plot(times[:len(perf_all_99['cache'])], perf_all_75['cache'], label=" 75%",
        color='green', linewidth=2, marker='o', markersize=4)
ax.plot(times[:len(perf_all_99['cache'])], perf_all_50['cache'], label=" 50%",
        color='blue', linewidth=2, marker='s', markersize=4)
ax.plot(times[:len(perf_all_99['cache'])], perf_all_25['cache'], label=" 25%",
        color='purple', linewidth=2, marker='*', markersize=8)
ax.legend(prop={'size': 10})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(y_lim[0], y_lim[1] * 1.05)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 11, va='top', ha='left')
myFmt = mdates.DateFormatter('%b, %d %H:%M')
ax.xaxis.set_major_formatter(myFmt)
#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.35)
#plt.gcf().autofmt_xdate()
#myFmt = mdates.DateFormatter('%H:%M')
#ax.xaxis.set_major_formatter(myFmt)

ax = axes[0, 1]
ax.set_title('video')
ax.xaxis_date()
ax.plot(times[:len(perf_all_99['video'])], perf_all_99['video'], label=" 99%", color='red', linewidth=2, marker='d', markersize=6)
ax.plot(times[:len(perf_all_99['video'])], perf_all_75['video'], label=" 75%", color='green', linewidth=2, marker='o', markersize=4)
ax.plot(times[:len(perf_all_99['video'])], perf_all_50['video'], label=" 50%", color='blue', linewidth=2, marker='s', markersize=4)
ax.plot(times[:len(perf_all_99['video'])], perf_all_25['video'], label=" 25%", color='purple', linewidth=2, marker='*', markersize=8)
ax.legend(prop={'size': 10})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(y_lim[0], 990)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 11, va='top', ha='left')
"""
ax.set_xlabel('Time', fontname="Arial", fontsize=11, weight='bold')
ax.set_ylabel('Benchmark Execution Time (Non-Nice User CPU Ticks)', fontname="Arial", fontsize=11, weight='bold')
"""
#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.35)
#plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
ax.xaxis.set_major_formatter(myFmt)

ax = axes[1, 0]
ax.set_title('nqueens')
ax.xaxis_date()
ax.plot(times[:len(perf_all_99['nqueens'])], perf_all_99['nqueens'], label=" 99%", color='red', linewidth=2, marker='d', markersize=6)
ax.plot(times[:len(perf_all_99['nqueens'])], perf_all_75['nqueens'], label=" 75%", color='green', linewidth=2, marker='o', markersize=4)
ax.plot(times[:len(perf_all_99['nqueens'])], perf_all_50['nqueens'], label=" 50%", color='blue', linewidth=2, marker='s', markersize=4)
ax.plot(times[:len(perf_all_99['nqueens'])], perf_all_25['nqueens'], label=" 25%", color='purple', linewidth=2, marker='*', markersize=8)
ax.legend(prop={'size': 10})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.set_ylim(y_lim[0], y_lim[1] * 1.05)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 8, va='top', ha='left')
myFmt = mdates.DateFormatter('%b, %d %H:%M')
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('Time', fontname="Arial", fontsize=8, weight='bold')
#ax.set_ylabel('Benchmark Execution Time (Non-Nice User CPU Ticks)', fontname="Arial", fontsize=16)


#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.35)
plt.gcf().autofmt_xdate()

ax = axes[1, 1]
ax.set_title('net')
ax.xaxis_date()
ax.plot(times[:len(perf_all_99['net'])], perf_all_99['net'], label=" 99%", color='red', linewidth=2, marker='d', markersize=6)
ax.plot(times[:len(perf_all_99['net'])], perf_all_75['net'], label=" 75%", color='green', linewidth=2, marker='o', markersize=4)
ax.plot(times[:len(perf_all_99['net'])], perf_all_50['net'], label=" 50%", color='blue', linewidth=2, marker='s', markersize=4)
ax.plot(times[:len(perf_all_99['net'])], perf_all_25['net'], label=" 25%", color='purple', linewidth=2, marker='*', markersize=8)
ax.legend(prop={'size': 10})

y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
print y_lim
print x_lim
ax.set_ylim([0, 100])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(times[:len(lst)][::2])
ax.set_xticklabels(times[:len(lst)][::2], fontsize = 8, va='top', ha='left')

ax.set_xlabel('Time', fontname="Arial", fontsize=8, weight='bold')
#ax.set_ylabel('Benchmark Execution Time (Non-Nice User CPU Ticks)', fontname="Arial", fontsize=16)

#plt.xlim(x_lim[0], x_lim[1] *1.05)
#plt.ylim(y_lim[0], y_lim[1] *1.35)
#plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%b, %d %H:%M')
ax.xaxis.set_major_formatter(myFmt)
#fig.tight_layout()

fig.text(0.06, 0.5, 'Benchmark Execution Time (Non-Nice User CPU Ticks)', va='center', rotation='vertical')

fig.savefig("total_performance.eps", bbox_inches='tight', format='eps')
plt.clf()
