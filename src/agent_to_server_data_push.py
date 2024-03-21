from opensearchpy import OpenSearch
from datetime import datetime
import json
import os


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
        verify_certs=True,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        ca_certs=ca_certs_path,
    )

    index_name = 'trial_agent_score_25'

    

    # Load your dataset_ratings from JSON
    path_json = os.getcwd()
    path_json = os.path.join(path_json, "anomaly_scores.json")
    with open(path_json, 'r') as file:
        data = json.load(file)

    # Iterate over each entry in the JSON data
    for entry in data:
        # Convert '@timestamp' to datetime object
        timestamp_dt = datetime.utcnow()

        # Extract epoch timestamp from datetime object
        timestamp_epoch = timestamp_dt.timestamp()

        # Create document with converted timestamp and other fields as floats
        document = {
            '@timestamp': timestamp_epoch,
            'ip': entry['ip'],
            'pid': entry['pid'],
            'anomaly_score': entry['anomaly_score']
        }

        # Push data to the index
        response = client.index(
            index=index_name,
            body=document,
            refresh=True
        )

        print(f'\nAdding document:')
        print(response)

