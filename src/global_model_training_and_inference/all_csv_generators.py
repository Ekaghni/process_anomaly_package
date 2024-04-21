import csv
import random
from datetime import datetime, timedelta


class CsvGenerator:

    def csv_generator_for_cpu_memory_training():
        def generate_data(rows):
            data = []
            for i in range(1, rows + 1):
                cpu_usage = random.randint(1, 100)
                memory_usage = random.randint(1, 100)
                signal = 'yes' if cpu_usage > 10 or memory_usage > 10 else 'no'
                data.append([i, cpu_usage, memory_usage, signal])
            return data
        def write_to_csv(data, filename='output.csv'):
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['count', 'cpu_usage', 'memory_usage', 'signal'])
                writer.writerows(data)
        total_rows = 100000 + 60000
        data = generate_data(total_rows)
        write_to_csv(data)


    def csv_generator_for_common_time_series_analysis():
        def generate_random_values(hour):
            if hour == 9: 
                cpu = round(random.uniform(0.7, 1), 2)
                memory = round(random.uniform(0.7, 1), 2)
            elif hour==6:
                cpu = round(random.uniform(0.7, 1), 2)
                memory = round(random.uniform(0.7, 1), 2)
            elif hour==13:
                cpu = round(random.uniform(0.7, 1), 2)
                memory = round(random.uniform(0.7, 1), 2)
            else:
                cpu = round(random.uniform(0, 1), 2)
                memory = round(random.uniform(0, 1), 2)
            return cpu, memory
        def generate_timestamps(date):
            timestamps = []
            current_time = datetime.strptime(date, '%Y-%m-%d')
            for _ in range(1440):
                timestamps.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))
                current_time += timedelta(minutes=1)
            return timestamps
        def generate_csv():
            start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')
            end_date = datetime.strptime('2024-12-31', '%Y-%m-%d')
            current_date = start_date
            with open('cpu_memory_usage.csv', 'w', newline='') as csvfile:
                fieldnames = ['Cpu_usage', 'Memory_usage', 'Timestamp_of_each_day']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                while current_date <= end_date:
                    timestamps = generate_timestamps(current_date.strftime('%Y-%m-%d'))
                    for timestamp in timestamps:
                        hour = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').hour
                        cpu, memory = generate_random_values(hour)
                        writer.writerow({'Cpu_usage': cpu, 'Memory_usage': memory, 'Timestamp_of_each_day': timestamp})
                    current_date += timedelta(days=1)
        generate_csv()


CsvGenerator.csv_generator_for_cpu_memory_training()
