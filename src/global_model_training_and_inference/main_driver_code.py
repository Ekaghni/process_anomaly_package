import sqlite3
from datetime import datetime, timedelta
from all_model_trainer_and_inference import ModelTrainerCPU
import psutil
import time

def connect_to_database():
    conn = sqlite3.connect('/home/telaverge/process_anomaly_package/src/global_model_training_and_inference/timestamps.db')
    return conn

def create_table(conn):
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS timestamps
                 (timestamp TEXT)''')
    conn.commit()

def get_last_timestamp(conn):
    c = conn.cursor()
    c.execute('''SELECT timestamp FROM timestamps ORDER BY timestamp DESC LIMIT 1''')
    row = c.fetchone()
    if row:
        return datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
    else:
        return None

def store_timestamp(conn, timestamp):
    c = conn.cursor()
    c.execute("INSERT INTO timestamps VALUES (?)", (timestamp,))
    conn.commit()

def check_elapsed_days(last_timestamp, current_timestamp):
    if last_timestamp is None:
        print("Time check res started just now..")
        return True
    # print("Time check res-------------> ",(current_timestamp - last_timestamp) >= timedelta(days=5))
    return (current_timestamp - last_timestamp) >= timedelta(days=30)

def main():
    # interval = 1
    # while True:

        conn = connect_to_database()
        create_table(conn)
        
        current_timestamp = datetime.now()
        last_timestamp = get_last_timestamp(conn)
        # print("last timestamp----> ",last_timestamp)
        # print("current timestamp----> ",current_timestamp)

        
        if (check_elapsed_days(last_timestamp, current_timestamp))==True:
            print("Inside condition 11111111")
            store_timestamp(conn, current_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            ModelTrainerCPU.train_common_time_series_analysis_lstm()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print("Timestamp----------> ",timestamp)
        # timestamp = "2024-12-30 06:45:00"
        result_from_lstm = ModelTrainerCPU.inference_common_time_series_analysis_lstm(timestamp_to_check = timestamp)
        # print("result_from_lstm---------------> ",result_from_lstm)
        if(result_from_lstm=="no"):
            current_cpu_usage= (int)(psutil.cpu_percent())
            current_memory_usage= (int)(psutil.virtual_memory().percent)
            result_from_cpu_memory_inference = ModelTrainerCPU.inference_cpu_memory_usage_random_forest_classifier(cpu_usage=current_cpu_usage, memory_usage=current_memory_usage)
            # print("Current cpu usage----> ",current_cpu_usage)
            # print("Current memory usage ----> ",current_memory_usage)
            # print("Result from Cpu Memory model----> ",result_from_cpu_memory_inference)
            if(result_from_cpu_memory_inference=="yes" and (current_cpu_usage> 20 or current_memory_usage > 20)):
                print("yes",flush=True)
            else:
                print("no",flush=True)
        else:
            print("no",flush=True)

        conn.close()
        # time.sleep(interval)

if __name__ == "__main__":
    main()
