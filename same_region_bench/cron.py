from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess

def run_bench():
    print ("Starting run...")
    subprocess.check_output(['python', 'bench.py'])

scheduler = BlockingScheduler()
scheduler.add_job(run_bench, 'interval', minutes=120, start_date='2020-05-22 10:00:00', end_date='2020-04-07 04:00:00')
scheduler.start()

