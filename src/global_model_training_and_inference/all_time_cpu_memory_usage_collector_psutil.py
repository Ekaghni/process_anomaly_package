import psutil
import csv
from datetime import datetime
import time
import os

def monitor_cpu_memory(interval=60, duration=None):
    file_exists = os.path.isfile('/home/telaverge/process_anomaly_package/src/global_model_training_and_inference/cpu_memory_usage.csv')
    with open('/home/telaverge/process_anomaly_package/src/global_model_training_and_inference/cpu_memory_usage.csv', mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['cpu_usage', 'memory_usage', 'Timestamp_of_each_day'])
        
        if duration:
            end_time = time.time() + duration
            while time.time() < end_time:
                write_to_csv(writer, file)
                time.sleep(interval)
                
        else:
            while True:
                write_to_csv(writer, file)
                time.sleep(interval)

def write_to_csv(writer, file):
    cpu_percent = round(psutil.cpu_percent() / 100, 2)
    memory_percent = round(psutil.virtual_memory().percent / 100, 2)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    writer.writerow([cpu_percent, memory_percent, timestamp])
    file.flush()

monitor_cpu_memory()
