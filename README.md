# CSCI6711-Project-Group6
Final Project on ML intrusion detection for CSCI6711


## How to run

### Step 1: Download the dataset
Full dataset available here: http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/

To run ML models, download as many of the Merged csv files as your machine can handle.
Recommenation is to begin with one file and scale up as desired.
Direct link to merged csv files: http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/CSV/MERGED_CSV/

To run snort, download PCAPs for the attacks you would like to replay
Direct link to PCAP files: http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/PCAP/

### Step 2: Run the experiments

To perform the experiments, locate the lines in the code that indicate the source file location, and change it to point to where you have downloaded the dataset.
Jupyter notebook files are intended to be run with Google CoLab. They will function with the free version.

NSGA2 implementation required too much RAM for Google CoLab free version, so it is provided in .py format. At least 16GB of RAM is recommended to run this file.

To run the Snort experiments on a linux machine, first install snort with:
```sudo apt install snort```

Then create a dummy local interface and assign it an ip address with the following comands:
```sudo ip link add dummy0 type dummy
sudo ip addr add 192.168.1.1/24 dev dummy0
sudo ip link set dummy0 up```

Run Snort on the interface:
```sudo snort -i dummy0 -c /etc/snort/snort.conf```

Then replay the PCAP files on the interface:
```sudo tcpreplay --intf1=dummy0 your_capture.pcap```


