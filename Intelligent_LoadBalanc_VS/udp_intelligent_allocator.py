import socket
import time
import random
import threading
import os
import numpy as np
import tensorflow as tf  # Import TensorFlow directly
from influxdb_client import InfluxDBClient

# List of Nomad client IP addresses and ports
CLIENTS = [
    ('192.168.56.5', 5005),
    ('192.168.56.6', 5005),
    ('192.168.56.7', 5005),
]

# Mapping of IPs to hostnames (as used in InfluxDB tags)
HOST_MAPPING = {
    '192.168.56.5': 'nomad-client1',
    '192.168.56.6': 'nomad-client2',
    '192.168.56.7': 'nomad-client3'
}

# Load the trained RL model
model = tf.keras.models.load_model('model/load_balancer_MAINmodel.keras')

# InfluxDB configuration
influx_url = "http://192.168.56.4:8086"
token = "YourAPIToken"
org = "YourOrgName"
bucket = "YourBucketName"

# Function to check if a client is reachable using ping
def is_client_up(ip):
    # Command for Linux
    response = os.system(f"ping -c 1 {ip} > /dev/null 2>&1")
    return response == 0

# Function to get current stats from InfluxDB for each client
def get_client_stats():
    stats = []
    with InfluxDBClient(url=influx_url, token=token, org=org) as client:
        query_api = client.query_api()
        for ip, _ in CLIENTS:
            host_tag = HOST_MAPPING[ip]  # Use the correct host tag
            query = f'''
            from(bucket: "{bucket}") 
            |> range(start: -1m) 
            |> filter(fn: (r) => r["_measurement"] == "cpu" or r["_measurement"] == "mem" or r["_measurement"] == "net")
            |> filter(fn: (r) => r["host"] == "{host_tag}") 
            |> last()
            '''
            result = query_api.query(org=org, query=query)
            
            # Initialize variables to store stats
            cpu_usage = None
            mem_usage = None
            net_usage = None
            
            # Process the result to extract CPU, memory, and network usage
            for table in result:
                for record in table.records:
                    measurement = record.get_measurement()
                    field = record.get_field()
                    value = record.get_value()
                    
                    if measurement == 'cpu' and field == 'usage_system':  # Adjust field names as necessary
                        cpu_usage = value  # Adjust to match correct usage field
                    elif measurement == 'mem' and field == 'used_percent':
                        mem_usage = value
                    elif measurement == 'net' and field == 'bytes_recv':
                        net_usage = value
            
            # Ensure stats are valid numbers; if not, default to zero
            cpu_usage = cpu_usage if cpu_usage is not None else 0
            mem_usage = mem_usage if mem_usage is not None else 0
            net_usage = net_usage if net_usage is not None else 0
            
            stats.append([cpu_usage, mem_usage, net_usage])
            
            # Print summarized stats for the client
            print(f"Client {ip} - CPU Usage: {cpu_usage:.2f}, Mem Usage: {mem_usage:.2f}%, Net Usage: {net_usage:.2f} bytes")
    
    return np.array(stats)

# Function to generate UDP traffic based on RL model's decision
def generate_udp_traffic():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    while True:
        stats = get_client_stats()
        print(f"Stats data: {stats}")  # Add logging for stats data
        
        # Flatten the stats to pass into the model
        input_data = stats.flatten().reshape(1, -1)
        # Predict the best client to send traffic to
        predictions = model.predict(input_data)
        predicted_action = np.argmax(predictions[0])
        
        # Log predictions for debugging
        print(f"Model predictions: {predictions[0]}, Chosen client: {predicted_action}")
        
        target_ip, target_port = CLIENTS[predicted_action]
        
        if is_client_up(target_ip):  # Check if the selected client is up
            message = bytearray(random.getrandbits(8) for _ in range(random.randint(10, 150)))
            try:
                sock.sendto(message, (target_ip, target_port))
                print(f"Sent {len(message)} bytes to {target_ip}:{target_port}")
            except Exception as e:
                print(f"Error sending data to {target_ip}:{target_port}: {e}")
            time.sleep(random.uniform(0.5, 5))  # Random delay
        else:
            print(f"{target_ip} is down. Skipping sending traffic.")
            time.sleep(10)  # Wait before checking again if the client is up

# Function to start generating UDP traffic
def start_traffic():
    thread = threading.Thread(target=generate_udp_traffic)
    thread.daemon = True
    thread.start()
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping UDP traffic generation.")

if __name__ == "__main__":
    print("Starting UDP traffic generator with RL model integration...")
    start_traffic()
