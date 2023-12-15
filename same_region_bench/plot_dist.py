import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import sys
from os import listdir
from os.path import isfile, join
sns.set(color_codes=True)

mypath = str(sys.argv[1])
files = sorted([f for f in listdir(mypath) if isfile(join(mypath, f))])
#files = [sys.argv[1], sys.argv[2], sys.argv[3]]
for f in files:
    with open(mypath + str(f), 'r') as opened_f:
        val = opened_f.read()
        values = filter(lambda x: x != '' ,val.split("\n"))
        print values
        values = np.array(values).astype(np.float)
        print values
        plot = sns.distplot(values, hist=False, kde=True, kde_kws={"lw": 3, "label": str(f)},)


#plt.ylim(0, 0.00035)
#plt.xlim(0, 30)

#plt.legend(loc='center right', bbox_to_anchor=(4.25, 0.5), ncol=10)
plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot.figure.savefig("output.png", bbox_inches='tight')

