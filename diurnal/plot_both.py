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
from matplotlib.dates import AutoDateLocator
from scipy import stats
from datetime import datetime, timedelta

def moving_average(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

us_east_cpu_dates = []
us_east_cpu_perf = []

ap_ne_cpu_dates = []
ap_ne_cpu_perf = []

us_east_net_dates = []
us_east_net_perf = []

ap_ne_net_dates = []
ap_ne_net_perf = []

temp = []
with open("us-east-1-oct2-oct4.csv", 'r') as opened_f:
    val = opened_f.read()
    values = filter(lambda x: x != '', val.split("\n"))
    counter = 0
    for x in values:
        if counter != 0:
            splt = x.split(",")
            #datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
            # ['2019-10-02 21:24:00.000,17202.6775']
            datetime_object = datetime.strptime(splt[0], '%Y-%m-%d %H:%M:%S.%f')
            temp.append((datetime_object, float(splt[1])))
        counter += 1

temp = sorted(temp)
for x, y in temp:
    us_east_cpu_dates.append(x)
    us_east_cpu_perf.append(y)

temp = []
with open("ap-northeast-2-oct-2-oct4.csv", 'r') as opened_f:
    val = opened_f.read()
    values = filter(lambda x: x != '', val.split("\n"))
    counter = 0
    for x in values:
        if counter != 0:
            splt = x.split(",")
            #datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
            # ['2019-10-02 21:24:00.000,17202.6775']
            datetime_object = datetime.strptime(splt[0], '%Y-%m-%d %H:%M:%S.%f')
            temp.append((datetime_object, float(splt[1])))
        counter += 1

temp = sorted(temp)
for x, y in temp:
    ap_ne_cpu_dates.append(x)
    ap_ne_cpu_perf.append(y)

temp = []
with open("us-east-1-oct2-oct4-net.csv", 'r') as opened_f:
    val = opened_f.read()
    values = filter(lambda x: x != '', val.split("\n"))
    counter = 0
    for x in values:
        if counter != 0:
            splt = x.split(",")
            #datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
            # ['2019-10-02 21:24:00.000,17202.6775']
            datetime_object = datetime.strptime(splt[0], '%Y-%m-%d %H:%M:%S.%f')
            temp.append((datetime_object, float(splt[1])))
        counter += 1

temp = sorted(temp)
for x, y in temp:
    us_east_net_dates.append(x)
    us_east_net_perf.append(y)

temp = []
with open("ap-northeast-2-oct2-oct4-net.csv", 'r') as opened_f:
    val = opened_f.read()
    values = filter(lambda x: x != '', val.split("\n"))
    counter = 0
    for x in values:
        if counter != 0:
            splt = x.split(",")
            #datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
            # ['2019-10-02 21:24:00.000,17202.6775']
            datetime_object = datetime.strptime(splt[0], '%Y-%m-%d %H:%M:%S.%f')
            temp.append((datetime_object, float(splt[1])))
        counter += 1

temp = sorted(temp)
for x, y in temp:
    ap_ne_net_dates.append(x)
    ap_ne_net_perf.append(y)


start = datetime(2019, 10, 02, 8, 00)
end = datetime(2019, 10, 04, 00, 00)

ticks = [start + timedelta(hours=i) for i in range(0,56,4)]
print ticks

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

fig.autofmt_xdate()

for ax in axes.flat:
    ax.set(xlabel='Time', ylabel='Benchmark Execution Time (ms)')
    #myFmt = mdates.DateFormatter('%b, %d')
    #locator = AutoDateLocator()
    #ax.xaxis.set_major_locator(locator)
    #ax.xaxis.set_major_formatter(myFmt)

ax = axes[0]
ax.set_title('Cache Benchmark Performance')
ax.scatter(us_east_cpu_dates, us_east_cpu_perf, color='blue', marker='d', label='us-east-1')
ax.plot(us_east_cpu_dates[9:-9], moving_average(us_east_cpu_perf, 12)[9:-9], color='black', linewidth=3, label="60 Minute Moving Average (us-east-1)")
ax.scatter(ap_ne_cpu_dates, ap_ne_cpu_perf, color='red', marker='*', label='ap-northeast-2')
ax.plot(ap_ne_cpu_dates[9:-9], moving_average(ap_ne_cpu_perf, 12)[9:-9], color='green', linewidth=3, label="60 Minute Moving Average (ap-northeast-2)")
ax.legend(prop={'size': 10})
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlim([ap_ne_cpu_dates[0], ap_ne_cpu_dates[-1]])
ax.set_xticks(ticks[3:len(ap_ne_cpu_dates)][::2])
ax.set_xticklabels(ticks[3:len(ap_ne_cpu_dates)][::2], fontsize = 8, va='top', ha='left')
ax.set_ylim(14000, 26500)
myFmt = mdates.DateFormatter('%b, %d, %a')
#locator = AutoDateLocator()
#ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(myFmt)

ax = axes[1]
ax.set_title('Net Benchmark Performance')

ax.scatter(us_east_net_dates, us_east_net_perf, color='blue', marker='d', label='us-east-1')
ax.scatter(ap_ne_net_dates, ap_ne_net_perf, color='red', marker='*', label='ap-northeast-2')
ax.legend(prop={'size': 10})
ax.set_xlim([ap_ne_net_dates[0], ap_ne_net_dates[-1]])
ax.set_xticks(ticks[3:len(ap_ne_net_dates) - 4][::2])
ax.set_xticklabels(ticks[3:len(ap_ne_cpu_dates)][::2], fontsize = 8, va='top', ha='left')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
myFmt = mdates.DateFormatter('%b, %d %a')
#locator = AutoDateLocator()
#ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(myFmt)
plt.savefig("combined_diurnal.eps", bbox_inches='tight', format='eps')
plt.clf()


print ("network variation within the same region")
print (np.max(us_east_net_perf) - np.min(us_east_net_perf)) / (np.min(us_east_net_perf)) * 100
print (np.max(ap_ne_net_perf) - np.min(ap_ne_net_perf)) / (np.min(ap_ne_net_perf)) * 100

total_net = np.concatenate((us_east_net_perf, ap_ne_net_perf))
print ("cross region")
print np.max(total_net)
print np.min(total_net)
print (np.max(total_net) - np.min(total_net)) / np.min(total_net) * 100


total_net = np.concatenate((us_east_cpu_perf, ap_ne_cpu_perf))
print ("cross region cpu")
print np.average(us_east_cpu_perf)
print np.average(ap_ne_cpu_perf)


difference = []
for x in xrange(len(us_east_cpu_perf[4:])):
    difference.append(np.abs(us_east_cpu_perf[x] - ap_ne_cpu_perf[x]) /  ap_ne_cpu_perf[x] * 100)

print np.average(difference)

print (np.average(us_east_cpu_perf) - np.average(ap_ne_cpu_perf)) / np.average(ap_ne_cpu_perf) * 100


