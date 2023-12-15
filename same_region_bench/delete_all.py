import subprocess
import json
from functools import partial
from multiprocessing.pool import ThreadPool

regions = ['us-east-1', 'eu-west-1', 'ap-northeast-2', 'us-east-2', 'us-west-1', 'us-west-2']

def delete_fn_parallel(obj):
    name = obj[0]['FunctionName']
    #name = "bench0"
    region = obj[1]
    print ('Deleting: ' + name + ' from region:\t' + str(region) + '\n')    
    try:
        subprocess.check_output(['aws', 'lambda', 'delete-function', '--function-name', name, '--region', str(region)])
    except:
        print ("already deleted...")

threadpool = ThreadPool(8)

for r in regions:
    print ("Deleting from region:\t" + str(r))
    output = subprocess.check_output(['./listall.sh', r])
    parsed_output = json.loads(output)
    
    test = [r for x in xrange(len(parsed_output['Functions']))]
    #test = [r for x in xrange(1)]
    
    zipped_obj = zip(parsed_output['Functions'], test)
    #zipped_obj = zip(test, test)
    
    parallel_delete = partial(delete_fn_parallel)
    threadpool.map(parallel_delete, zipped_obj)

