import socket
import time
import random
import threading
import os

# List of Nomad client IP addresses and ports
CLIENTS = [
    ('192.168.56.5', 5005),
    ('192.168.56.6', 5005),
    ('192.168.56.7', 5005),
]

# Function to check if a client is reachable using ping
def is_client_up(ip):
    # Use '-n 1' on Windows to send a single ping request
    response = os.system(f"ping -n 1 {ip} >nul")
    return response == 0

# Function to generate random UDP traffic
def generate_udp_traffic(target_ip, target_port):
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    while True:
        if is_client_up(target_ip):  # Check if the client is up
            # Generate random message with random length
            message = bytearray(random.getrandbits(8) for _ in range(random.randint(10, 150)))
            
            # Send the message to the target
            try:
                sock.sendto(message, (target_ip, target_port))
                print(f"Sent {len(message)} bytes to {target_ip}:{target_port}")
            except Exception as e:
                print(f"Error sending data to {target_ip}:{target_port}: {e}")
            
            # Wait for a random interval before sending the next packet
            time.sleep(random.uniform(0.5, 5))  # Random delay between 0.5 to 5 seconds
        else:
            print(f"{target_ip} is down. Skipping sending traffic.")
            time.sleep(10)  # Wait before checking again if the client is up

# Function to start generating UDP traffic for each client in a separate thread
def start_traffic():
    threads = []
    for ip, port in CLIENTS:
        thread = threading.Thread(target=generate_udp_traffic, args=(ip, port))
        thread.daemon = True
        threads.append(thread)
        thread.start()
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping UDP traffic generation.")

if __name__ == "__main__":
    print("Starting UDP traffic generator...")
    start_traffic()

