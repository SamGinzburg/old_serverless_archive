from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool
import subprocess
import json
import base64
import re
import random
from functools import partial
from collections import defaultdict
import time
import os
import numpy as np
import matplotlib.pyplot as plt


NUM_REPEAT_INVOCATIONS = 5

def delete_fn_parallel(obj):
    name = obj[0]
    region = obj[1]
    #print ('Deleting: ' + name + ' from region:\t' + str(region) + '\n')    
    try:
        subprocess.check_output(['aws', 'lambda', 'delete-function', '--function-name', name, '--region', str(region)])
    except:
        print ("already deleted...")


def create_fn(params):
    name = params[0]
    fn_memory = params[1]
    q = params[2]
    region = params[3]
    benchmark = params[4]

    create_output = subprocess.check_output(['aws', 'lambda', 'create-function',
                                             '--function-name', name,
                                             '--zip-file', 'fileb://' + benchmark + '/function.zip',
                                             '--handler', 'lambda_test.handler',
                                             '--runtime', 'nodejs12.x', '--role',
                                             '', # aws arn goes here
                                             '--timeout', '300', '--memory-size', fn_memory, '--region', region])
    #print (create_output)
    if 'error' in create_output:
        q.put([])
        return
    else:
        print ("created probe fn:\t" + name)
    
    # aws lambda put-function-concurrency --function-name my-function --reserved-concurrent-executions 100
    #subprocess.check_output(['aws', 'lambda', 'put-function-concurrency', '--function-name', name,
    #                         '--reserved-concurrent-executions', '100'])

    ctx = []
    total_bill = []
    total_bench = []

    ctx_switches = []
    interrupts = []
    adjusted_interrupts = []
    stolen_ticks = []
    stolen_percent_lst = []
    benchmark_times = []
    err_cnt = 0
    for x in xrange(NUM_REPEAT_INVOCATIONS):
        invoke_output = subprocess.check_output(['aws', 'lambda', 'invoke',
                                                 '--function-name', name,
                                                 '--log-type', 'Tail',
                                                 '--region', region,
                                                 '--payload',
                                                 '{"probe":"true"}',
                                                 'outputfile.txt'])

       
        parsed_output = json.loads(invoke_output)
        #print parsed_output
        log = parsed_output['LogResult']
        decoded_log = base64.b64decode(log)

        #print (decoded_log)
        # get the boot time
        #uptime = re.findall(r"uptime\(.+\n.+\n.+\n.+\n.+\nbtime\ (.*)", decoded_log)
        # TODO: fix
        uptime = ['0']

        context_switches = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*)\ CPU", decoded_log)

        context_switches2 = re.findall(r"instanceid2\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*)\ CPU", decoded_log)


        # get stolen ticks, non-nice user ticks, system ticks, idle ticks, io-wait ticks, IRQ cpu ticks, and softirq cpu ticks
        # we use this info to compute % ticks stolen
        instanceid = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.\ +(.*?)\ stolen", decoded_log)
        instanceid2 = re.findall(r"instanceid2\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.\ +(.*?)\ stolen", decoded_log)

        non_nice_user = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.\ +(.*?)\ non", decoded_log)
        non_nice_user2 = re.findall(r"instanceid2\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.\ +(.*?)\ non", decoded_log)

        system = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ system", decoded_log)
        system2 = re.findall(r"instanceid2\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ system", decoded_log)

        idle_cpu = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ idle", decoded_log)
        idle_cpu2 = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ idle", decoded_log)

        io_wait = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ IO", decoded_log)
        io_wait2 = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ IO", decoded_log)

        irq = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ IRQ", decoded_log)
        irq2 = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ IRQ", decoded_log)

        softirq = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ softirq", decoded_log)
        softirq2 = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ softirq", decoded_log)

        interrupts1 = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.\ +(.*?)\ interrupts", decoded_log)

        interrupts2 = re.findall(r"instanceid2\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.\ +(.*?)\ interrupts", decoded_log)

        timer_cpu0 = re.findall(r"interrupts1l1\((\d+)\)interrupts", decoded_log)
        timer_cpu1 = re.findall(r"interrupts1l2\((\d+)\)interrupts", decoded_log)
        after_timer_cpu0 = re.findall(r"interrupts2l1\((\d+)\)interrupts", decoded_log)
        after_timer_cpu1 = re.findall(r"interrupts2l2\((\d+)\)interrupts", decoded_log)

        benchmark_time = re.findall(r"benchmark:\s(.*?)ms", decoded_log)

        bill = re.findall(r"Billed\ Duration: (.*?)\ ", decoded_log)
        try:
            diff0 = float(after_timer_cpu0[0]) - float(timer_cpu0[0])
            diff1 = float(after_timer_cpu1[0]) - float(timer_cpu1[0])
            ctx_switches.append((float(context_switches2[0]) - float(context_switches[0])))
            interrupts.append((float(interrupts2[0]) - float(interrupts1[0])))
            adjusted_interrupts.append((float(interrupts2[0]) - float(interrupts1[0])) - diff0 - diff1)
            stolen_ticks.append((float(instanceid2[0]) - float(instanceid[0])))
            total_bill.append(int(bill[0]))
            total_bench.append((float(non_nice_user2[0]) - float(non_nice_user[0])))


            stolenticks = (float(instanceid2[0]) - float(instanceid[0]))
            userticks = (float(non_nice_user2[0]) - float(non_nice_user[0]))
            systemticks = (float(system2[0]) - float(system[0]))
            idleticks = (float(idle_cpu2[0]) - float(idle_cpu[0]))
            ioticks = (float(io_wait[0]) - float(io_wait[0]))
            irqticks = (float(irq2[0]) - float(irq[0]))
            softirqticks = (float(softirq2[0]) - float(softirq[0]))

            stolen_percent = (stolenticks) / (stolenticks + userticks + systemticks + idleticks + ioticks + irqticks + softirqticks)
            stolen_percent = stolen_percent * 100
            stolen_percent_lst.append(stolen_percent)

            benchmark_times.append(float(benchmark_time[0]))

        except:
            print ("err in fn")
            err_cnt += 1

    try:
        first_bench = total_bench[0]
        raw_perf_first = benchmark_times[0]
        first_stolen = stolen_ticks[0]
        print ([uptime[0], name, total_bill,
                total_bench, ctx_switches, interrupts, adjusted_interrupts,
                stolen_ticks, first_bench, first_stolen, stolen_percent_lst, raw_perf_first, benchmark_times])

        for x in xrange(len(total_bench)):
            print ("loop count:\t" + str(x))
            q.put([uptime[0], name, sum(total_bill),
                   total_bench[x], ctx_switches[x], interrupts[x], adjusted_interrupts[x],
                   stolen_ticks[x], first_bench, first_stolen, stolen_percent_lst[x],
                   benchmark_times[x], raw_perf_first, time.time()])
    except:
        print ("ERROR - placing dummy data to continue exec...")
    for x in xrange(err_cnt):
        q.put([])
        print ("placing dummy data to continue exec...")

def invoke_fn(params):
    name = params[0]
    invocation_count = int(params[1])
    q = params[2]
    region = params[3]
    benchmark = params[4]

    for count in xrange(invocation_count):
        print ("Invoking:\t" + str(name) + "\tcount:\t" + str(count))
        invocation_time = int(time.time())
        invoke_output = subprocess.check_output(['aws', 'lambda', 'invoke',
                                                 '--region', region,
                                                 '--function-name', name, '--log-type', 'Tail',
                                                 '--payload',
                                                 '{"probe":"true"}',
                                                 'outputfile.txt'])

       
        parsed_output = json.loads(invoke_output)
        log = parsed_output['LogResult']
        decoded_log = base64.b64decode(log)
        bill = re.findall(r"Billed\ Duration: (.*?)\ ", decoded_log)
        
        instanceid = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.\ +(.*?)\ stolen", decoded_log)
        instanceid2 = re.findall(r"instanceid2\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.\ +(.*?)\ stolen", decoded_log)

        non_nice_user = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.\ +(.*?)\ non", decoded_log)
        non_nice_user2 = re.findall(r"instanceid2\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.\ +(.*?)\ non", decoded_log)

        system = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ system", decoded_log)
        system2 = re.findall(r"instanceid2\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ system", decoded_log)

        idle_cpu = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ idle", decoded_log)
        idle_cpu2 = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ idle", decoded_log)

        io_wait = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ IO", decoded_log)
        io_wait2 = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ IO", decoded_log)

        irq = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ IRQ", decoded_log)
        irq2 = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ IRQ", decoded_log)

        softirq = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ softirq", decoded_log)
        softirq2 = re.findall(r"instanceid\(.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n.+\n\ +(.*?)\ softirq", decoded_log)

        benchmark_time = re.findall(r"benchmark:\s(.*?)ms", decoded_log)
        #print ([name, bill, invocation_time, instanceid, instanceid2, non_nice_user, non_nice_user2, system,
        #        system2, idle_cpu, idle_cpu2, io_wait, io_wait2, irq, irq2, softirq, softirq2])
        try:
            stolenticks = (float(instanceid2[0]) - float(instanceid[0]))
            userticks = (float(non_nice_user2[0]) - float(non_nice_user[0]))
            systemticks = (float(system2[0]) - float(system[0]))
            idleticks = (float(idle_cpu2[0]) - float(idle_cpu[0]))
            ioticks = (float(io_wait2[0]) - float(io_wait[0]))
            irqticks = (float(irq2[0]) - float(irq[0]))
            softirqticks = (float(softirq2[0]) - float(softirq[0]))

            stolen_percent = (stolenticks) / (stolenticks + userticks + systemticks + idleticks + ioticks + irqticks + softirqticks)
            stolen_percent = stolen_percent * 100
            raw_perf = float(benchmark_time[0])
            q.put([name, bill[0], invocation_time, stolen_percent, userticks, raw_perf])
        except IndexError as ie:
            print (str(ie))
            q.put([])


def create_functions_with_probe(threadpool, fn_memory, num_fns, region, benchmark):
    queue = Queue()

    params = [["probe" + str(x), str(fn_memory), queue, region, benchmark] for x in range(num_fns)]

    threadpool.map(create_fn, params)

    temp = []
    try:
        while len(temp) < num_fns * NUM_REPEAT_INVOCATIONS:
            val = queue.get(True, timeout=60)
            if val != []:
                temp.append(val)
            print ("received:\t"+str(len(temp)))
    except:
        print ("Error receiving from queue, continuing...")
    return temp

def probe_fn(threadpool, fn_names, invocation_count, region, benchmark):
    queue = Queue()

    params = [[str(name), str(invocation_count), queue, region, benchmark] for name in fn_names]

    threadpool.map(invoke_fn, params)
    
    temp = []
    while len(temp) < (len(fn_names) * invocation_count):
        temp.append(queue.get(True))

    return temp

def select_from_most_colocated(d, num_to_select):
    ret_list = []

    for key, lst in sorted(d.items(), key=lambda x: len(x[1]), reverse=True)[:num_to_select]:
        ret_list.append(lst[random.randint(0, len(lst) - 1)])

    return ret_list

def select_from_least_colocated(d, num_to_select):
    ret_list = []

    for key, lst in sorted(d.items(), key=lambda x: len(x[1]), reverse=False)[:num_to_select]:
        ret_list.append(lst[random.randint(0, len(lst) - 1)])

    return ret_list

def select_from_least_ctx_switches(d, num_to_select):
    return sorted(d, key=lambda x: x[2])[:num_to_select]


def select_from_most_ctx_switches(d, num_to_select):
    return sorted(d, key=lambda x: x[2], reverse=True)[:num_to_select]


def names_from_lst(lst):
    return [x[1] for x in lst]


def dump_ctx_dist_to_file(d, region, directory):
    with open(directory + "/" + str(region) + "_" + str(int(time.time())) + ".txt", 'a+') as f:
        for item in d:
            f.write(str(item[2]) + "\n")

if __name__ == '__main__':

    threadpool = ThreadPool(6)

    #regions = ['us-east-1', 'ap-northeast-2', 'us-west-1']
    #regions = ['ap-northeast-2']
    regions = ['us-east-1']
    benchmarks = ['cache', 'video', 'nqueens', 'net']
    #benchmarks = ['video']
    #benchmarks = ['cache', 'video']
    run_dir = str(int(time.time()))

    fn_results = []

    for region in regions:
        for benchmark in benchmarks:
            output = subprocess.check_output(['./listall.sh', region])
            parsed_output = json.loads(output)
            test = [str(x['FunctionName']) for x in parsed_output['Functions']]
            params = [[str(name), region] for name in test]
            threadpool.map(delete_fn_parallel, params)

            print ("Arbitrage for region:\t" + str(region))
            # first create a bunch of functions
            unique_fns = create_functions_with_probe(threadpool, 2048, 50, region, benchmark)
            # filter all non-errored out functions
            unique_fns = filter(lambda x: len(x) != 0, unique_fns)

            fn_results.append((unique_fns, region))
            #dump_ctx_dist_to_file(unique_fns, region)

            new_lst = []
            d = defaultdict(list)
            search_bill = 0

            ctx_switch_lst = []
            interrupts_lst = []
            interrupts_adj_lst = []
            stolen_ticks_lst = []
            bill_lst = []
            bench_lst = []
            first_bench = []
            first_stolen = []
            names_lst = []
            print unique_fns
            
            """
            q.put([uptime[0], name, sum(total_bill),
                       total_bench.pop(0), ctx_switches.pop(0), interrupts.pop(0), adjusted_interrupts.pop(0),
                       stolen_ticks.pop(0)])
            """


            fn_stolen_ticks = defaultdict(int)
            fn_bench = defaultdict(int)
            stolen_ratio_lst = []
            benchmark_final_times = []
            unix_timestamps = []
            first_benchmark = []
            for terms in unique_fns:
                element = terms[0]
                name = terms[1]

                bill = terms[2]
                bench = terms[3]
                ctx_switches = terms[4]
                interrupts = terms[5]
                adjusted_ints = terms[6]
                stolen_ticks = terms[7]
                first_bnch = terms[8]
                first_stln = terms[9]
                stolen_ratio_percent = terms[10]
                bt = terms[11]
                first_bt = terms[12]
                uts = terms[13]

                dist = (int(element) % 100)

                interrupts_lst.append(interrupts)
                interrupts_adj_lst.append(adjusted_ints)
                stolen_ticks_lst.append(stolen_ticks)
                ctx_switch_lst.append(ctx_switches)
                bill_lst.append(bill)
                bench_lst.append(bench)
                first_stolen.append(first_stln)
                first_bench.append(first_bnch)
                names_lst.append(name)
                stolen_ratio_lst.append(stolen_ratio_percent)
                benchmark_final_times.append(bt)
                unix_timestamps.append(uts)
                first_benchmark.append(first_bt)

                fn_stolen_ticks[name] = max(stolen_ratio_percent, fn_stolen_ticks[name])
                fn_bench[name] += bench

                search_bill += bill


            # now oversample from all functions that had an extremely high amount of CPU stealing
            print list(fn_stolen_ticks.items())
            temp = [[x, y] for x, y in list(fn_stolen_ticks.items())]
            print temp
            repeat_sample_fns = sorted(temp, key=lambda l: l[1], reverse=True)[:min(3, len(unique_fns))]
            print repeat_sample_fns
            names = [n[0] for n in repeat_sample_fns]
            print names
            results = probe_fn(threadpool, names, NUM_REPEAT_INVOCATIONS * 3, region, benchmark)
            results = filter(lambda x: len(x) != 0, results)

            print ("Overallocation results:\t" + str(results))

            directory = "ctx_correlation/" + run_dir + "/" + str(benchmark)
            if not os.path.exists(directory):
                os.makedirs(directory)

            if not os.path.exists(directory + "/oversample/"):
                os.makedirs(directory + "/oversample/")

            with open(directory + "/oversample/" + str(region) + ".txt", 'a+') as f_out:
                for row in results:
                    # bench \t stolen
                    f_out.write(str(row[3]) + "\t")
                    f_out.write(str(row[4]) + "\n")

            # write down the bill / num CTX's
            directory = "ctx_correlation/" + run_dir + "/" + str(benchmark)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(directory + "/" + str(region) + ".txt", 'a+') as f_out:
                for a, b, c, d, e, f, g, h, i, j, k, l, m in zip(ctx_switch_lst, interrupts_lst, interrupts_adj_lst,
                                                     stolen_ticks_lst, bench_lst, bill_lst, first_bench,
                                                     first_stolen, names_lst, stolen_ratio_lst,
                                                     benchmark_final_times, first_benchmark, unix_timestamps):
                    f_out.write(str(a) + "\t")
                    f_out.write(str(b) + "\t")
                    f_out.write(str(c) + "\t")
                    f_out.write(str(d) + "\t")
                    f_out.write(str(e) + "\t")
                    f_out.write(str(f) + "\t")
                    f_out.write(str(g) + "\t")
                    f_out.write(str(h) + "\t")
                    f_out.write(str(i) + "\t")
                    f_out.write(str(j) + "\t")
                    f_out.write(str(k) + "\t")
                    f_out.write(str(l) + "\t")
                    f_out.write(str(m) + "\n")

            # unneeded
            """
            directory = "ctx_dists/" + run_dir + "/" + str(benchmark) + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(directory + "/" + str(region) + ".txt", 'a+') as f_out:
                for a, b, c, d, e, f, g, h, i, j in zip(ctx_switch_lst, interrupts_lst, interrupts_adj_lst,
                                                     stolen_ticks_lst, bench_lst, bill_lst, first_bench,
                                                     first_stolen, names_lst, stolen_ratio_lst):
            """

            output = subprocess.check_output(['./listall.sh', region])
            parsed_output = json.loads(output)
            test = [str(x['FunctionName']) for x in parsed_output['Functions']]
            params = [[str(name), region] for name in test]
            threadpool.map(delete_fn_parallel, params)

