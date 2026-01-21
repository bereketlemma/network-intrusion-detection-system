# 🛡️ Network Intrusion Detection System (NIDS)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A real-time Machine Learning-based Network Intrusion Detection System that analyzes network traffic and identifies malicious activities with 95% F1-score accuracy.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance) • [Project Structure](#-project-structure)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **Network Intrusion Detection System** using Machine Learning to classify network traffic as normal or malicious. The system can detect **7 different types of network attacks** in real-time and provides an interactive dashboard for monitoring and visualization.

### Key Highlights:
- 🎯 **95% F1-Score** on attack detection
- 📊 **2.5M+ network packets** analyzed from CICIDS2017 dataset
- 🔍 **7 attack types** detected (DoS, DDoS, Port Scanning, Brute Force, Web Attacks, Bots)
- 📈 **Interactive Dashboard** with real-time monitoring simulation
- ⚡ **Fast predictions** - 40,000 samples in 0.11 seconds

---

## ✨ Features

### 🤖 Machine Learning Model
- **Random Forest Classifier** with 100 decision trees
- Handles class imbalance using undersampling
- Feature scaling with StandardScaler
- 52 network flow features analyzed

### 📊 Interactive Dashboard
- **Real-time traffic monitoring** simulation
- **Attack distribution** visualization (pie charts, bar charts)
- **Model performance metrics** display
- **Feature importance** analysis
- **Live prediction testing** on individual samples

### 🔒 Attack Detection Capabilities
| Attack Type | Description | Detection Rate |
|-------------|-------------|----------------|
| Normal Traffic | Legitimate network activity | 92% |
| DoS | Denial of Service attacks | 95% |
| DDoS | Distributed DoS attacks | 100% |
| Port Scanning | Network reconnaissance | 99% |
| Brute Force | Password cracking attempts | 97% |
| Web Attacks | SQL injection, XSS, etc. | 100% |
| Bots | Botnet traffic | 100% |

---

## 🎬 Demo

### Dashboard Screenshots

#### 📊 Security Overview Dashboard
- Attack distribution pie chart
- Attack type counts bar chart
- Summary statistics table

#### 🔴 Live Network Traffic Monitor
- Real-time traffic feed with alerts
- Green boxes for normal traffic
- Red boxes for detected attacks
- Threat level indicator

#### 📈 Model Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- Per-class performance bar chart
- Detailed metrics table

#### 🔍 Feature Importance Analysis
- Top 15 most important features
- Feature descriptions
- Importance scores

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/NIDS.git
cd NIDS
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
1. Go to [Kaggle - CICIDS2017 Cleaned Dataset](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/data)
2. Download the CSV file
3. Place it in the `data/` folder as `cicids2017_cleaned.csv`

---

## 💻 Usage

### Option 1: Run Complete Pipeline

```bash
# Step 1: Train the model
python src/model_training.py

# Step 2: Launch dashboard
streamlit run dashboard/app.py
```

### Option 2: Run Individual Modules

```bash
# Test data loader
python src/data_loader.py

# Test feature engineering
python src/feature_engineering.py

# Train model
python src/model_training.py
```

### Option 3: Use Jupyter Notebook
```bash
jupyter notebook notebooks/NIDS_Complete_Walkthrough.ipynb
```

### Accessing the Dashboard
After running `streamlit run dashboard/app.py`, open your browser to:
```
http://localhost:8501
```

---

## 📁 Dataset

### CICIDS2017: Cleaned & Preprocessed

| Property | Value |
|----------|-------|
| **Source** | Canadian Institute for Cybersecurity |
| **Total Samples** | 2,520,751 |
| **Features** | 52 numeric features |
| **Classes** | 7 (1 normal + 6 attack types) |
| **File Size** | ~685 MB |

### Attack Distribution
```
Normal Traffic    2,095,057  (83.1%)
DoS                 193,745  (7.7%)
DDoS                128,014  (5.1%)
Port Scanning        90,694  (3.6%)
Brute Force           9,150  (0.36%)
Web Attacks           2,143  (0.085%)
Bots                  1,948  (0.077%)
```

### Key Features Used
1. **Destination Port** - Target port number
2. **Flow Duration** - Duration of network flow
3. **Packet Length Mean** - Average packet size
4. **Flow Bytes/s** - Bytes per second
5. **Fwd Packet Length Max** - Maximum forward packet size
6. And 47 more network flow features...

---


### Top 5 Most Important Features

1. **Destination Port** (10.72%)
2. **Init_Win_bytes_backward** (4.62%)
3. **Packet Length Mean** (4.55%)
4. **Max Packet Length** (4.09%)
5. **Subflow Fwd Bytes** (4.09%)

---

## 🔧 Technical Details

### Data Preprocessing Pipeline

1. **Data Cleaning**: Remove duplicates, handle infinite/NaN values
2. **Label Encoding**: Convert attack names to numeric labels
3. **Train/Test Split**: 80% training, 20% testing (stratified)
4. **Feature Scaling**: StandardScaler (mean=0, std=1)
5. **Class Balancing**: RandomUnderSampler for imbalanced classes

### Model Architecture
```python
RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    max_depth=None,        # No depth limit
    class_weight='balanced',
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
```

---

## 🙏 Acknowledgments

- **Canadian Institute for Cybersecurity** for the CICIDS2017 dataset
- **Kaggle** for hosting the cleaned dataset
- **Scikit-learn** team for the amazing ML library
- **Streamlit** team for the dashboard framework

---

