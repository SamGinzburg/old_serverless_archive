import subprocess
import json
import base64
import re
import numpy as np
import time

times = []
workload = []

retries = 0
x = 0
while x < 100:
    print ("Attempting to run function:\t" + str(x))
    output = subprocess.check_output(['./invoke.sh', 'test2', r'{"bench":"false","loopcnt":"32768"}', 'us-east-1'])
    parsed_output = json.loads(output)
    log = parsed_output['LogResult']
    decoded_log = base64.b64decode(log)
    print (decoded_log)
    n = re.findall(r"benchmark:\ (.*)ms", decoded_log)
    m = re.findall(r"Billed\ Duration: (.*?)\ ", decoded_log)
    try:
        # if skipped don't increment counter
        if len(n) > 0:
            x += 1
            workload.append(n[0])
        else:
            retries += 1
        print (m[0])
        times.append(int(m[0]))
    except:
        pass

print ("Final report")
print ("Total end to end execution time:\t" + str(sum(times)))
print ("Retries:\t" + str(retries))

for x in xrange(len(times)):
    with open('us-east-1-times_nostrat.txt', 'a+') as f:
            f.write(str(times[x]) + "\n")

for x in xrange(len(workload)):
    with open('us-east-1-execution_times_nostrat.txt', 'a+') as f:
        f.write(str(workload[x]) + "\n")

