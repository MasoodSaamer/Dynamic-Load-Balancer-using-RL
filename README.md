# Dynamic Load Balancer using Reinforcement Learning

This project implements a **dynamic load balancer** for network traffic using **Reinforcement Learning (RL)**. It simulates a set of Nomad clients and allocates UDP traffic among them based on their resource usage (CPU, memory, and network traffic) to minimize latency. The project utilizes **Nomad** for task orchestration, **InfluxDB** for storing system metrics, and **Telegraf** for collecting those metrics. The RL model learns to distribute traffic intelligently to balance client load and minimize latency.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Vagrant Configuration](#vagrant-configuration)
- [UDP Intelligent Allocator](#udp-intelligent-allocator)
- [InfluxDB Setup](#influxdb-setup)
- [Testing](#testing)

## Overview
The **Dynamic Load Balancer** uses **Reinforcement Learning** to allocate traffic to Nomad clients by continuously monitoring their CPU, memory, and network usage. The RL model dynamically learns the optimal allocation strategy that minimizes client latency while preventing overloading any single client.

This project was tested on a local environment using **Vagrant** to create the Nomad server and client VMs, **InfluxDB** to store the resource metrics, and **Telegraf** to collect metrics from each VM.

## Architecture
1. **Nomad**: Orchestrates the clients and manages the infrastructure.
2. **InfluxDB**: Stores client metrics such as CPU, memory, and network usage.
3. **Telegraf**: Collects the resource metrics from each Nomad client and sends them to InfluxDB.
4. **Reinforcement Learning Model**: A Keras-based RL model that dynamically adjusts the load balancing strategy.
5. **UDP Traffic Generator**: Simulates network traffic between the Nomad clients.

## Technologies Used
- **Python 3.10**: For developing the RL model and traffic allocator.
- **TensorFlow/Keras**: For building and training the RL model.
- **Vagrant**: For setting up the Nomad server and client VMs.
- **Nomad**: Used for task orchestration and managing client resources.
- **InfluxDB**: Used as the database to store client resource metrics.
- **Telegraf**: Collects system metrics from each client VM.
- **Ubuntu 18.04 (Bionic)**: Operating system for the VMs.
  

## Prerequisites
- **Vagrant**: Install from [Vagrant official site](https://www.vagrantup.com/downloads).
- **VirtualBox**: Install from [VirtualBox official site](https://www.virtualbox.org/).
- **Python 3.10**: Install from [Python official site](https://www.python.org/downloads/).

2. Set up the Vagrant VMs
The Vagrantfile in the repository sets up the Nomad server and client VMs. To spin up the VMs:

bash
Copy code
vagrant up
This command will:

Create and configure the Nomad server and three client VMs.
Install and configure Nomad, Consul, InfluxDB, and Telegraf on these VMs.
3. Set up Python environment
To set up your Python environment:

bash
Copy code
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # If requirements.txt is provided
Install the necessary dependencies manually if requirements.txt is missing:

bash
Copy code
pip install tensorflow gym numpy influxdb-client
4. Configure InfluxDB
After the Vagrant setup, you will need to configure InfluxDB using the provided command in the Vagrantfile:

bash
Copy code
influx setup --username admin --password 'admin123' --org YourOrgName --bucket YourBucketName --retention 0 --force --token "YourAPIToken"
Make sure to replace the placeholders (YourOrgName, YourBucketName, etc.) with your actual setup details.

Usage
Running the RL-based Load Balancer
To run the RL-based load balancer, execute the Python script:

bash
Copy code
python udp_intelligent_allocator.py
This script will:

Load the pre-trained RL model (load_balancer_MAINmodel.keras).
Continuously monitor the resource usage of each Nomad client by querying InfluxDB.
Dynamically allocate traffic to clients based on the RL model’s decision to balance load and minimize latency.
Training the RL Model
If you want to re-train the RL model, run the rl_model.py script:

bash
Copy code
python rl_model.py
This script simulates the Nomad clients’ resource usage and trains the RL model to make intelligent traffic allocation decisions.

Vagrant Configuration
The Vagrantfile in this repository configures the environment with:

1 Nomad Server VM (IP: 192.168.56.4).
3 Nomad Client VMs (IP Range: 192.168.56.5-192.168.56.7).
InfluxDB and Telegraf on all VMs for monitoring resource usage.
You can manage the VMs with:

bash
Copy code
vagrant up   # To start the VMs
vagrant halt # To stop the VMs
vagrant destroy # To remove the VMs
UDP Intelligent Allocator
The udp_intelligent_allocator.py script is the main component of this project. It:

Uses a pre-trained RL model to decide how to allocate UDP traffic among the Nomad clients.
Collects real-time resource usage metrics (CPU, Memory, Network) from InfluxDB.
Makes dynamic decisions to balance the load and avoid overloading any client.
InfluxDB Setup
The project uses InfluxDB 2.x to collect and store metrics from each Nomad client. You can access InfluxDB’s web interface at http://192.168.56.4:8086 (after starting the VMs).

Telegraf is responsible for sending the system metrics (CPU, memory, network) to InfluxDB from each client VM. These metrics are queried in the udp_intelligent_allocator.py script to inform the RL model.

Testing
Local Testing with Vagrant
Spin up the Vagrant environment using vagrant up.
SSH into the Nomad server or any of the clients using:
bash
Copy code
vagrant ssh nomad-server
vagrant ssh nomad-client1
Run the Python script locally:
bash
Copy code
python udp_intelligent_allocator.py
Monitor the traffic allocation decisions and resource usage on InfluxDB.
