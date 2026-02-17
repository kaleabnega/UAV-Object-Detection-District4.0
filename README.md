# UAV-Based Traffic Monitoring with Secure Data Transmission

## Overview

This project leverages a UAV (Unmanned Aerial Vehicle) to detect, count, and transmit vehicle data in real-time. Using on-device object detection and a secure MQTT communication channel with TLS, the system ensures low-latency, encrypted transmission of vehicle counts to a ground station dashboard.

This repository intends to shed light on a subset of our conference paper's focus, which is comprised of adaptive lightweight security for TinyML-driven UAV systems in smart cities, where we present the system architecture, quantized model optimization, and adaptive security evaluation in detail.

## Objectives

- Real-time vehicle detection using an edge-optimized YOLOv8 model (TFLite + INT8).
- Low-bandwidth communication using MQTT protocol.
- Secure transmission using TLS with fallback protocol switching for latency optimization.
- Dashboard visualization of traffic statistics using Streamlit.

## Key Features

- **On-Device Inference:** Object detection runs on UAV, reducing reliance on cloud infrastructure.
- **TLS Secured MQTT:** Data is transmitted via MQTT with TLS authentication and encryption.
- **Adaptive Security Layer:** System dynamically chooses between TLS and lightweight fallback mode.
- **Streamlit Dashboard:** Easy-to-use web dashboard to view traffic data in real-time.

## Paper

This project is based on the following publication:

**Title:** _Adaptive Lightweight Security for TinyML Driven Unmanned Aerial Vehicles in Smart Cities_
**Conference:** IEEE (2025)

The paper presents:

- Quantization of YOLOv8 to INT8 TFLite for resource-constrained UAV platforms.
- Experimental comparison of full TLS, AEAD (ChaCha20-Poly1305), and HMAC-only modes.
- Measured reductions in inference latency, memory footprint, communication delay, and CPU load through adaptive security switching.

**Paper Link:** [Link](https://ieeexplore.ieee.org/document/11291809)

## Citation

If you use this repository or build upon this work, please cite our paper:

```bibtex
@inproceedings{uav_tinyml_security_2025,
  title={Adaptive Lightweight Security for TinyML Driven Unmanned Aerial Vehicles in Smart Cities},
  author={Jamil, Norziana and Nega, Kaleab and Rozaidi, Muhammad Haikal and Kulkarni, Parag and Ramli, Ramona and Al-Ghaili, Abbas M.},
  booktitle={Proceedings of the 3rd International Conference on Cyber Resilience (ICCR)},
  address={Dubai, United Arab Emirates},
  year={2025}
}

```


## Security Architecture

We utilize:

- TLS v1.2 with Mosquitto broker for encrypted communication.
- Mutual certificate-based authentication.
- Adaptive fallback to authentication-only mode when resources are constrained (CPU/load/battery), reducing overhead.

## Folder Structure

```
UAV-Object-Detection-District4.0/
├── app.py # Streamlit dashboard
├── video_stream.py # Main object detection + MQTT code
├── yolov8n_saved_model/ # INT8 TFLite model
├── labels.txt # Class labels
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- `paho-mqtt`, `opencv-python`, `numpy`, `streamlit`, `tensorflow` (lite)

### 1. Install Dependencies

#### Installation & Setup

First, clone the repository and navigate into it:

```bash

git clone https://github.com/kaleabnega/UAV-Object-Detection-District4.0.git
cd UAV-Object-Detection-District4.0
```

#### Set up a virtual environment (recommended):

```bash

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

### 2. Run the Mosquitto Broker (locally)

Make sure mosquitto is installed and running with tls.conf configuration.

### 3. Run the Streamlit App

```bash
streamlit run app.py
```
