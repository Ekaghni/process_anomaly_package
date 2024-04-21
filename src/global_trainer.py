import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.ensemble import IsolationForest
import json
import socket
import time
import requests
import logging
from opensearchpy import OpenSearch
from datetime import datetime
import json
import os
import subprocess
import csv


def update_agent_ossec():

    OSSEC_CONF_FILE = "/var/ossec/etc/ossec.conf"
    NEW_BLOCK = """
    <localfile>
    <location>/home/telaverge/process_anomaly_package/src/example_1.log</location>
    <log_format>syslog</log_format>
    </localfile>
    """
    if os.path.exists(OSSEC_CONF_FILE) and os.access(OSSEC_CONF_FILE, os.W_OK):
        with open(OSSEC_CONF_FILE, "r") as file:
            lines = file.readlines()

        if NEW_BLOCK.strip() in "".join(lines).strip():
            print("New block already present in", OSSEC_CONF_FILE)
        else:

            last_localfile_line = None
            for i, line in enumerate(lines):
                if "</localfile>" in line:
                    last_localfile_line = i + 1

            if last_localfile_line is not None:
    
                lines[last_localfile_line:last_localfile_line] = [NEW_BLOCK + "\n"]
                print("New block added to", OSSEC_CONF_FILE)


            with open(OSSEC_CONF_FILE, "w") as file:
                file.writelines(lines)

        ###################################### ossec file for cpu memory addition ####################################

        block_to_add = """
        <!-- Log analysis -->
        <!-- CPU, memory, disk metric -->
        <!-- CPU, memory, disk metric -->
        <ossec_config>

            <localfile>
                <log_format>full_command</log_format>
                <command>
                    if [ "$(sudo python3 /home/telaverge/process_anomaly_package/src/global_model_training_and_inference/main_driver_code.py)" = "yes" ]; then
                        cpu_output="$(top -bn1 | grep Cpu | awk '{print $2+$4+$6+$12+$14+$16}')"
                    else
                        cpu_output="0"
                    fi

                    if [ "$(sudo python3 /home/telaverge/process_anomaly_package/src/global_model_training_and_inference/main_driver_code.py)" = "yes" ]; then
                        mem_output="$(free -m | awk 'NR==2{printf "%.2f", $3*100/$2 }')"
                    else
                        mem_output="0"
                    fi

                    disk_output="$(df -h | awk '$NF=="/"{print $5}'|sed 's/%//g')"

                    echo "$cpu_output $mem_output $disk_output"
                </command>
                <alias>general_health_metrics</alias>
                <out_format>$(timestamp) $(hostname) general_health_check: $(log)</out_format>
                <frequency>50</frequency>
            </localfile>
            
            
                <!-- load average metrics -->
            <localfile>
                <log_format>full_command</log_format>
                <command>if [ "$(sudo python3 /home/telaverge/process_anomaly_package/src/global_model_training_and_inference/main_driver_code.py)" = "yes" ]; then uptime | grep load | awk '{print $(NF-2),$(NF-1),$NF}' | sed 's/\,\([0-9]\{1,2\}\)/.\1/g'; fi</command>
                <alias>load_average_metrics</alias>
                <out_format>$(timestamp) $(hostname) load_average_check: $(log)</out_format>
                <frequency>50</frequency>
            </localfile>


            <!-- memory metrics -->
            <localfile>
                <log_format>full_command</log_format>
                <command>if [ "$(sudo python3 /home/telaverge/process_anomaly_package/src/global_model_training_and_inference/main_driver_code.py)" = "yes" ]; then free --bytes| awk 'NR==2{print $3,$7}'; fi</command>
                <alias>memory_metrics</alias>
                <out_format>$(timestamp) $(hostname) memory_check: $(log)</out_format>
                <frequency>50</frequency>
            </localfile>

            <!-- disk metrics -->
            <localfile>
                <log_format>full_command</log_format>
                <command>if [ "$(sudo python3 /home/telaverge/process_anomaly_package/src/global_model_training_and_inference/main_driver_code.py)" = "yes" ]; then df -B1 | awk '$NF=="/"{print $3,$4}'; fi</command>
                <alias>disk_metrics</alias>
                <out_format>$(timestamp) $(hostname) disk_check: $(log)</out_format>
                <frequency>50</frequency>
            </localfile>

        </ossec_config>
        """

        file_path = "/var/ossec/etc/ossec.conf"

        with open(file_path, 'r') as file:
            file_content = file.read()
            if block_to_add.strip() not in file_content:

                with open(file_path, 'a') as file:
                    file.write(block_to_add)
                    print("Block added successfully.")
            else:
                print("Block already exists in the file.")

    ##############################################################################################################################################

    else:
        print("Could not access", OSSEC_CONF_FILE)

update_agent_ossec()

subprocess.run(["sudo", "bash", "/home/telaverge/process_anomaly_package/src/ossec_update.sh"])



def get_ip_address():
        try:
            
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
            return ip_address
        except Exception as e:
            print("Error:", e)
            return None
class ProcessAnomalyDetector:
    def __init__(self, input_size=10, learning_rate=0.000001, num_epochs=400, batch_size=64):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None

    def collect_process_data(self):
        processes = []
        pids = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'open_files',
                                         'create_time', 'cmdline', 'environ', 'connections', 'cwd', 'num_threads',
                                         'status']):
            if not all(key in proc.info for key in ['cpu_percent', 'memory_percent', 'open_files',
                                                    'create_time', 'cmdline', 'environ', 'connections',
                                                    'cwd', 'num_threads', 'status']):
                continue
            if proc.info['cmdline'] is not None and proc.info['open_files'] is not None:
                processes.append([
                    proc.info['cpu_percent'],
                    proc.info['memory_percent'],
                    len(proc.info['open_files']),
                    proc.info['create_time'],
                    len(proc.info['cmdline']),
                    len(proc.info['environ']),
                    len(proc.info['connections']),
                    len(proc.info['cwd']),
                    proc.info['num_threads'],
                    1 if proc.info['status'] == 'running' else 0
                ])
                pids.append(proc.pid)
        print("Collected PIDs:", pids)
        return processes, pids

    def preprocess_data(self, processes):
        return torch.tensor(processes, dtype=torch.float)

    def train_model(self, processed_data):
        self.model = AnomalyDetectionModel(self.input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(processed_data, processed_data)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for data, _ in data_loader:
                optimizer.zero_grad()
                reconstructions = self.model(data)
                loss = criterion(reconstructions, data)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss}")
        torch.save(self.model.state_dict(), 'anomaly_detection_model.pth')

    def get_ip_address(self):
        try:
            # print("ip address method called")
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            # print(ip_address)
            s.close()
            return ip_address
        except Exception as e:
            print("Error:", e)
            return None

    def infer_anomaly_scores(self, processed_data, pids, model_path='anomaly_detection_model.pth',
                             output_path='anomaly_scores.json'):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        anomaly_scores = []
        with torch.no_grad():
            for data in processed_data:
                reconstruction = self.model(data.unsqueeze(0))
                error = torch.mean(torch.square(data - reconstruction))
                anomaly_scores.append(error.item())

        max_error = max(anomaly_scores)
        min_error = min(anomaly_scores)
        normalized_scores = [(error - min_error) / (max_error - min_error) for error in anomaly_scores]
        ip = self.get_ip_address()
        pid_anomaly_scores = [{"ip": ip, "pid": pid, "anomaly_score": score} for pid, score in
                              zip(pids, normalized_scores)]

        with open(output_path, 'w') as f:
            json.dump(pid_anomaly_scores, f, indent=4)

        print("Anomaly scores saved to:", output_path)
        return pid_anomaly_scores

    def run(self):
        processes, pids = self.collect_process_data()
        processed_data = self.preprocess_data(processes)
        self.train_model(processed_data)
        anomaly_scores = self.infer_anomaly_scores(processed_data, pids)
        return anomaly_scores


class AnomalyDetectionModel(nn.Module):
    def __init__(self, input_size):
        super(AnomalyDetectionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




def data_pusher():
    host = "172.16.10.131"
    port = 8200
    auth = ("admin", "admin")
    current_directory = os.getcwd()
    current_directory = os.path.join(current_directory, "trial_cert.pem")
    ca_certs_path = current_directory

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        ca_certs=ca_certs_path,
    )

    index_name = 'trial_agent_score_25'

    

    
    path_json = os.getcwd()
    path_json = os.path.join(path_json, "anomaly_scores.json")
    with open(path_json, 'r') as file:
        data = json.load(file)

    for entry in data:

        timestamp_dt = datetime.utcnow()


        timestamp_epoch = timestamp_dt.timestamp()


        document = {
            '@timestamp': timestamp_epoch,
            'ip': entry['ip'],
            'pid': entry['pid'],
            'anomaly_score': entry['anomaly_score']
        }


        response = client.index(
            index=index_name,
            body=document,
            refresh=True
        )

        print(f'\nAdding document:')
        print(response)



def get_process_info():
    process_info = []
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'open_files',
                                         'create_time', 'cmdline', 'environ', 'connections', 'cwd', 'num_threads',
                                         'status'])
            process_info.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return process_info


def rate_anomaly(cpu_percent, mem_percent, num_threads, open_files, connections):
    max_anomaly_score = 1
    min_anomaly_score = 0

    anomaly_score = (cpu_percent + mem_percent + num_threads + open_files + connections) / (5 * 100)

    normalized_score = (anomaly_score - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    return np.clip(normalized_score, 0, 1)


def process_weight_rate():
    processes = get_process_info()

    anomaly_scores = []
    for p in processes:
        cpu_percent = p['cpu_percent']
        mem_percent = p['memory_percent']
        num_threads = p['num_threads']
        open_files = len(p['open_files'])
        connections = len(p['connections'])
        ip = get_ip_address()
        anomaly_rate = rate_anomaly(cpu_percent, mem_percent, num_threads, open_files, connections)
        anomaly_scores.append({"ip": ip,"pid": p['pid'], "anomaly_score": anomaly_rate})
        print(f"PID: {p['pid']}, Name: {p['name']}, Username: {p['username']}, Anomaly rate: {anomaly_rate:.2f}")

    return anomaly_scores


class ProcessAnomalyDetection:

    @staticmethod
    def get_process_info():
        process_info = []
        for proc in psutil.process_iter():
            try:
                pinfo = proc.as_dict(attrs=['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads',
                                            'open_files', 'connections'])
                process_info.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return process_info
    
    @staticmethod
    def get_ip_address():
        try:
            # print("ip address method called")
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            # print(ip_address)
            s.close()
            return ip_address
        except Exception as e:
            print("Error:", e)
            return None

    @staticmethod
    def preprocess_data(processes):
        data = []
        for p in processes:
            data.append([p['cpu_percent'], p['memory_percent'], p['num_threads'], len(p['open_files']),
                         len(p['connections'])])
        return np.array(data)

    @staticmethod
    def train_anomaly_detector(data):
        clf = IsolationForest(contamination='auto', random_state=42)
        clf.fit(data)
        return clf


    @staticmethod
    def get_anomaly_scores(clf, processes):
        anomaly_scores = clf.decision_function(processes)
        return anomaly_scores

    @staticmethod
    def save_anomaly_scores(processes, anomaly_scores, filename='anomaly_scores.json'):
        anomaly_data = []
        ip = ProcessAnomalyDetection.get_ip_address()
        for idx, p in enumerate(processes):
            
            pid = p['pid']
            anomaly_score = anomaly_scores[idx]
            anomaly_data.append({"ip": ip, "pid": pid, "anomaly_score": anomaly_score})

        with open(filename, 'w') as file:
            json.dump(anomaly_data, file, indent=4)

    @staticmethod
    def main_runner():
        processes = ProcessAnomalyDetection.get_process_info()
        processed_data = ProcessAnomalyDetection.preprocess_data(processes)
        anomaly_detector = ProcessAnomalyDetection.train_anomaly_detector(processed_data)
        anomaly_scores = ProcessAnomalyDetection.get_anomaly_scores(anomaly_detector, processed_data)
        ip = ProcessAnomalyDetection.get_ip_address()

        ann = []
        for idx, p in enumerate(processes):
            
            pid = p['pid']
            name = p['name']
            anomaly_score = anomaly_scores[idx]
            ann.append({"ip": ip,"pid": p['pid'], "anomaly_score": anomaly_score})
            

        ProcessAnomalyDetection.save_anomaly_scores(processes, anomaly_scores)

        print("annnn-----------> ",ann)
        print("\n\n")

        return ann

def monitor_cpu_memory(interval=60, duration=None):
    file_exists = os.path.isfile('/home/telaverge/process_anomaly_package/src/global_model_training_and_inference/cpu_memory_usage.csv')
    with open('/home/telaverge/process_anomaly_package/src/global_model_training_and_inference/cpu_memory_usage.csv', mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['cpu_usage', 'memory_usage', 'Timestamp_of_each_day'])
        
        # if duration:
        #     end_time = time.time() + duration
        #     while time.time() < end_time:
        #         write_to_csv(writer, file)
        #         time.sleep(interval)
                
        else:
            # while True:
            write_to_csv(writer, file)
            # time.sleep(interval)

def write_to_csv(writer, file):
    cpu_percent = round(psutil.cpu_percent() / 100, 2)
    memory_percent = round(psutil.virtual_memory().percent / 100, 2)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    writer.writerow([cpu_percent, memory_percent, timestamp])

if __name__ == "__main__":


    while True:


        
        current_directory = os.getcwd()
        current_directory = os.path.join(current_directory, "file.log")

        logging.basicConfig(filename=current_directory, level=logging.INFO)
        logging.info('Reached step 1')



        exclude_list = ["wazuh", "kworker", "elastic","nvidia", "ubuntu"]

        detector = ProcessAnomalyDetector()
        anomaly_scores_global = detector.run()

        anomaly_scores_weight = process_weight_rate()

        anomaly_scores_trial = ProcessAnomalyDetection.main_runner()

        pids_global = {entry['pid'] for entry in anomaly_scores_global}
        pids_weight = {entry['pid'] for entry in anomaly_scores_weight}
        pids_trial = {entry['pid'] for entry in anomaly_scores_trial}
        common_pids = pids_global.intersection(pids_weight, pids_trial)
        anomaly_scores_global_filtered = [entry for entry in anomaly_scores_global if entry['pid'] in common_pids]
        anomaly_scores_weight_filtered = [entry for entry in anomaly_scores_weight if entry['pid'] in common_pids]
        anomaly_scores_trial_filtered = [entry for entry in anomaly_scores_trial if entry['pid'] in common_pids]

        pid_score_mapping = {}
        for pid in common_pids:
            pid_score_mapping[pid] = []

        for entry in anomaly_scores_global_filtered:
            pid_score_mapping[entry['pid']].append(entry['anomaly_score'])

        for entry in anomaly_scores_weight_filtered:
            pid_score_mapping[entry['pid']].append(entry['anomaly_score'])

        for entry in anomaly_scores_trial_filtered:
            pid_score_mapping[entry['pid']].append(entry['anomaly_score'])

        avg_scores = {pid: np.mean(scores) for pid, scores in pid_score_mapping.items()}
        logging.info('Reached step 2....')

        print("\nAverage anomaly scores:")
        filtered_pid_anomaly_scores = []
        for pid, score in avg_scores.items():
            try:
                process_name = psutil.Process(pid).name()
                if any(string in process_name for string in exclude_list):
                    print(f"Excluding Process Name: {process_name}, PID: {pid}, Average Score: {score:.2f}")
                else:
                    print(f"Process Name: {process_name}, PID: {pid}, Average Score: {score:.2f}")
                    filtered_pid_anomaly_scores.append({"ip": get_ip_address(), "pid": pid, "anomaly_score": score})
            except psutil.NoSuchProcess:
                print(f"PID: {pid}, Average Score: {score:.2f} (Process not found)")



        with open('/home/telaverge/process_anomaly_package/src/average_anomaly_scores.json', 'w') as f:
            json.dump(avg_scores, f, indent=4)
        print("Average anomaly scores saved to average_anomaly_scores.json")

        output_path = '/home/telaverge/process_anomaly_package/src/anomaly_scores.json'
        with open(output_path, 'w') as f:
            json.dump(filtered_pid_anomaly_scores, f, indent=4)
        print("Filtered anomaly scores saved to:", output_path)

        print("Average anomaly score len---> ", len(avg_scores))
        print("Filtered anomaly score len---> ", len(filtered_pid_anomaly_scores))

        logging.info('Reached step 3....')

        # data_pusher()

        with open('/home/telaverge/process_anomaly_package/src/anomaly_scores.json', 'r') as file:
            data = json.load(file)
        with open('/home/telaverge/process_anomaly_package/src/example_1.log', 'a') as logfile:
            for entry in data:

                logfile.write(json.dumps(entry) + '\n')
        counter =0

        while True:
            monitor_cpu_memory()
            print("Writing data to file......")

            time.sleep(60)
            counter +=1

            if(counter==10):
              break


        # time.sleep(600)




