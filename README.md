# Network Intrusion Detection System (NIDS)

A machine learning–based Network Intrusion Detection System that analyzes network traffic and detects malicious activity using the CICIDS2017 dataset.

## Overview

- Trained on **2.5M+ network flow records**
- Detects **7 attack categories** (DoS, DDoS, Port Scanning, Brute Force, Web Attacks, Bots, Normal)
- Built with a focus on **accuracy, interpretability, and scalability**
- Includes an **interactive Streamlit dashboard** for visualization and monitoring

## Key Features

- Random Forest–based classifier optimized for tabular network data  
- Handles severe **class imbalance** using undersampling  
- **97.47% accuracy** and **98.21% F1-score** on unseen test data  
- Feature importance analysis for security insights  
- Interactive dashboard for model performance and predictions  

## Dataset

- **CICIDS2017** (Canadian Institute for Cybersecurity)
- 52 numeric network flow features
- Highly imbalanced real-world network traffic data

## Tech Stack

- Python, Pandas, NumPy  
- Scikit-learn, Imbalanced-learn  
- Streamlit  

## Run the Project

```bash
# Train the full model
python train_full_model.py

# Launch the dashboard
streamlit run dashboard/app.py

```