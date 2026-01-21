"""
NIDS Interactive Dashboard
==========================
A Streamlit-based dashboard for visualizing network intrusion detection.

Features:
1. Real-time attack detection simulation
2. Attack distribution visualization
3. Model performance metrics
4. Feature importance charts
5. Live network traffic monitoring simulation

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import time
from datetime import datetime, timedelta
import random

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="NIDS - Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .attack-alert {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .normal-traffic {
        background-color: #00C851;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND PREPROCESSORS
# ============================================================
@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects."""
    models_dir = "models"
    
    try:
        model = joblib.load(os.path.join(models_dir, "random_forest_model.joblib"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        label_mapping = joblib.load(os.path.join(models_dir, "label_mapping.joblib"))
        feature_names = joblib.load(os.path.join(models_dir, "feature_names.joblib"))
        return model, scaler, label_mapping, feature_names, True
    except FileNotFoundError:
        return None, None, None, None, False

# ============================================================
# LOAD SAMPLE DATA
# ============================================================
@st.cache_data
def load_sample_data(n_samples=10000):
    """Load a sample of the dataset for visualization."""
    data_file = "data/cicids2017_cleaned.csv"
    try:
        df = pd.read_csv(data_file, nrows=n_samples)
        return df, True
    except FileNotFoundError:
        return None, False

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_attack_color(attack_type):
    """Return color based on attack type."""
    colors = {
        'Normal Traffic': '#00C851',
        'DoS': '#ff4444',
        'DDoS': '#CC0000',
        'Port Scanning': '#ffbb33',
        'Brute Force': '#ff8800',
        'Web Attacks': '#aa66cc',
        'Bots': '#0099CC'
    }
    return colors.get(attack_type, '#666666')

def simulate_network_traffic(n_packets=20):
    """Simulate network traffic data for live monitoring."""
    attack_types = ['Normal Traffic', 'DoS', 'DDoS', 'Port Scanning', 
                    'Brute Force', 'Web Attacks', 'Bots']
    
    # Weighted probabilities (mostly normal traffic)
    weights = [0.83, 0.07, 0.05, 0.03, 0.01, 0.005, 0.005]
    
    traffic = []
    base_time = datetime.now()
    
    for i in range(n_packets):
        packet = {
            'timestamp': base_time + timedelta(seconds=i*0.5),
            'src_ip': f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
            'dst_ip': f"10.0.{random.randint(1,255)}.{random.randint(1,255)}",
            'src_port': random.randint(1024, 65535),
            'dst_port': random.choice([80, 443, 22, 21, 3389, 8080, 3306]),
            'protocol': random.choice(['TCP', 'UDP', 'ICMP']),
            'bytes': random.randint(64, 65535),
            'attack_type': random.choices(attack_types, weights=weights)[0]
        }
        traffic.append(packet)
    
    return pd.DataFrame(traffic)

# ============================================================
# MAIN DASHBOARD
# ============================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Network Intrusion Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Real-time AI-powered network security monitoring</p>', 
                unsafe_allow_html=True)
    
    # Load model
    model, scaler, label_mapping, feature_names, model_loaded = load_model_and_preprocessors()
    
    # Load sample data
    df, data_loaded = load_sample_data()
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)
    st.sidebar.title("🎛️ Control Panel")
    
    page = st.sidebar.radio(
        "Navigate",
        ["📊 Dashboard", "🔴 Live Monitor", "📈 Model Performance", "🔍 Feature Analysis", "🎯 Test Prediction"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    if model_loaded:
        st.sidebar.success("✅ Model Loaded")
    else:
        st.sidebar.error("❌ Model Not Found")
        st.sidebar.info("Run `python src/model_training.py` first")
    
    if data_loaded:
        st.sidebar.success("✅ Data Loaded")
    else:
        st.sidebar.error("❌ Data Not Found")
    
    # ============================================================
    # PAGE: DASHBOARD
    # ============================================================
    if page == "📊 Dashboard":
        st.header("📊 Security Overview Dashboard")
        
        if data_loaded:
            # Top metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_packets = len(df)
            attack_count = len(df[df['Attack Type'] != 'Normal Traffic'])
            normal_count = len(df[df['Attack Type'] == 'Normal Traffic'])
            attack_rate = (attack_count / total_packets) * 100
            
            with col1:
                st.metric("📦 Total Packets", f"{total_packets:,}")
            with col2:
                st.metric("✅ Normal Traffic", f"{normal_count:,}")
            with col3:
                st.metric("🚨 Attacks Detected", f"{attack_count:,}")
            with col4:
                st.metric("⚠️ Attack Rate", f"{attack_rate:.2f}%")
            
            st.markdown("---")
            
            # Charts row
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Attack Distribution")
                attack_counts = df['Attack Type'].value_counts()
                
                fig = px.pie(
                    values=attack_counts.values,
                    names=attack_counts.index,
                    color=attack_counts.index,
                    color_discrete_map={
                        'Normal Traffic': '#00C851',
                        'DoS': '#ff4444',
                        'DDoS': '#CC0000',
                        'Port Scanning': '#ffbb33',
                        'Brute Force': '#ff8800',
                        'Web Attacks': '#aa66cc',
                        'Bots': '#0099CC'
                    },
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Attack Type Counts")
                
                fig = px.bar(
                    x=attack_counts.index,
                    y=attack_counts.values,
                    color=attack_counts.index,
                    color_discrete_map={
                        'Normal Traffic': '#00C851',
                        'DoS': '#ff4444',
                        'DDoS': '#CC0000',
                        'Port Scanning': '#ffbb33',
                        'Brute Force': '#ff8800',
                        'Web Attacks': '#aa66cc',
                        'Bots': '#0099CC'
                    }
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Attack Type",
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Attack details table
            st.subheader("📋 Attack Summary Table")
            summary = df['Attack Type'].value_counts().reset_index()
            summary.columns = ['Attack Type', 'Count']
            summary['Percentage'] = (summary['Count'] / total_packets * 100).round(2)
            summary['Risk Level'] = summary['Attack Type'].apply(
                lambda x: '🟢 Low' if x == 'Normal Traffic' else 
                         ('🔴 Critical' if x in ['DDoS', 'DoS'] else '🟡 Medium')
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Please load the dataset first")
    
    # ============================================================
    # PAGE: LIVE MONITOR
    # ============================================================
    elif page == "🔴 Live Monitor":
        st.header("🔴 Live Network Traffic Monitor")
        st.markdown("*Simulated real-time network packet analysis*")
        
        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)
        with col2:
            refresh_rate = st.slider("Refresh Rate (sec)", 1, 10, 3)
        with col3:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Simulate traffic
        traffic_df = simulate_network_traffic(30)
        
        # Live stats
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        attacks = traffic_df[traffic_df['attack_type'] != 'Normal Traffic']
        
        with col1:
            st.metric("📡 Packets/sec", len(traffic_df))
        with col2:
            st.metric("🚨 Threats Detected", len(attacks))
        with col3:
            st.metric("🌐 Unique Sources", traffic_df['src_ip'].nunique())
        with col4:
            threat_level = "🟢 LOW" if len(attacks) < 3 else ("🟡 MEDIUM" if len(attacks) < 7 else "🔴 HIGH")
            st.metric("⚠️ Threat Level", threat_level)
        
        st.markdown("---")
        
        # Live traffic feed
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📜 Live Traffic Feed")
            
            for _, packet in traffic_df.iterrows():
                is_attack = packet['attack_type'] != 'Normal Traffic'
                
                if is_attack:
                    st.markdown(f"""
                    <div class="attack-alert">
                        🚨 <b>ALERT:</b> {packet['attack_type']} detected!<br>
                        <small>{packet['timestamp'].strftime('%H:%M:%S')} | 
                        {packet['src_ip']}:{packet['src_port']} → 
                        {packet['dst_ip']}:{packet['dst_port']} | 
                        {packet['protocol']} | {packet['bytes']} bytes</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="normal-traffic">
                        ✅ Normal Traffic<br>
                        <small>{packet['timestamp'].strftime('%H:%M:%S')} | 
                        {packet['src_ip']}:{packet['src_port']} → 
                        {packet['dst_ip']}:{packet['dst_port']} | 
                        {packet['protocol']} | {packet['bytes']} bytes</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🎯 Attack Types Detected")
            attack_summary = traffic_df['attack_type'].value_counts()
            
            fig = px.pie(
                values=attack_summary.values,
                names=attack_summary.index,
                color=attack_summary.index,
                color_discrete_map={
                    'Normal Traffic': '#00C851',
                    'DoS': '#ff4444',
                    'DDoS': '#CC0000',
                    'Port Scanning': '#ffbb33',
                    'Brute Force': '#ff8800',
                    'Web Attacks': '#aa66cc',
                    'Bots': '#0099CC'
                }
            )
            fig.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
    
    # ============================================================
    # PAGE: MODEL PERFORMANCE
    # ============================================================
    elif page == "📈 Model Performance":
        st.header("📈 Model Performance Metrics")
        
        if model_loaded:
            # Model info
            st.subheader("🤖 Model Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**Model Type:** Random Forest")
            with col2:
                st.info(f"**Number of Trees:** {model.n_estimators}")
            with col3:
                st.info(f"**Features Used:** {len(feature_names)}")
            
            st.markdown("---")
            
            # Performance metrics (from training)
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Accuracy", "93.13%", "Target: 95%")
            with col2:
                st.metric("📌 Precision", "97.56%", "+2.56%")
            with col3:
                st.metric("🔍 Recall", "93.13%", "-1.87%")
            with col4:
                st.metric("⚖️ F1-Score", "95.06%", "✅ Target Met")
            
            st.markdown("---")
            
            # Per-class performance
            st.subheader("📋 Per-Class Performance")
            
            performance_data = {
                'Attack Type': ['Bots', 'Brute Force', 'DDoS', 'DoS', 'Normal Traffic', 'Port Scanning', 'Web Attacks'],
                'Precision': [0.04, 0.48, 0.99, 0.79, 1.00, 0.95, 0.04],
                'Recall': [1.00, 0.97, 1.00, 0.95, 0.92, 0.99, 1.00],
                'F1-Score': [0.08, 0.64, 0.99, 0.86, 0.96, 0.97, 0.08],
                'Support': [33, 152, 2019, 3084, 33224, 1451, 37]
            }
            perf_df = pd.DataFrame(performance_data)
            
            # Heatmap-style visualization
            fig = go.Figure()
            
            for i, metric in enumerate(['Precision', 'Recall', 'F1-Score']):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=perf_df['Attack Type'],
                    y=perf_df[metric],
                    text=[f"{v:.0%}" for v in perf_df[metric]],
                    textposition='auto',
                ))
            
            fig.update_layout(
                barmode='group',
                height=400,
                xaxis_title="Attack Type",
                yaxis_title="Score",
                yaxis_range=[0, 1.1]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table view
            st.dataframe(perf_df.style.format({
                'Precision': '{:.0%}',
                'Recall': '{:.0%}',
                'F1-Score': '{:.0%}',
                'Support': '{:,}'
            }), use_container_width=True, hide_index=True)
            
        else:
            st.warning("⚠️ Model not loaded. Run training first.")
    
    # ============================================================
    # PAGE: FEATURE ANALYSIS
    # ============================================================
    elif page == "🔍 Feature Analysis":
        st.header("🔍 Feature Importance Analysis")
        
        if model_loaded:
            st.markdown("*Which network features are most important for detecting attacks?*")
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            # Top 15 features
            top_features = importance_df.tail(15)
            
            # Horizontal bar chart
            fig = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                height=500,
                title="Top 15 Most Important Features",
                xaxis_title="Importance Score",
                yaxis_title="Feature"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature descriptions
            st.subheader("📖 Feature Descriptions")
            
            feature_descriptions = {
                'Destination Port': 'Target port number - key indicator of attack type',
                'Init_Win_bytes_backward': 'TCP window size in backward direction',
                'Packet Length Mean': 'Average size of packets in the flow',
                'Max Packet Length': 'Largest packet in the flow',
                'Subflow Fwd Bytes': 'Total bytes sent in forward direction',
                'Total Length of Fwd Packets': 'Sum of all forward packet sizes',
                'Average Packet Size': 'Mean packet size across the flow',
                'Fwd Packet Length Mean': 'Average forward packet size',
                'Flow IAT Max': 'Maximum inter-arrival time between packets',
                'Flow Packets/s': 'Packet rate per second'
            }
            
            for feature in top_features.tail(10)['Feature'].values[::-1]:
                if feature in feature_descriptions:
                    st.markdown(f"**{feature}**: {feature_descriptions[feature]}")
        else:
            st.warning("⚠️ Model not loaded. Run training first.")
    
    # ============================================================
    # PAGE: TEST PREDICTION
    # ============================================================
    elif page == "🎯 Test Prediction":
        st.header("🎯 Test Attack Prediction")
        
        if model_loaded and data_loaded:
            st.markdown("*Select a sample from the dataset to see the model's prediction*")
            
            # Sample selector
            sample_idx = st.slider("Select Sample Index", 0, len(df)-1, 0)
            
            # Get sample
            sample = df.iloc[sample_idx]
            actual_label = sample['Attack Type']
            
            # Prepare features for prediction
            feature_values = sample[feature_names].values.reshape(1, -1)
            scaled_features = scaler.transform(feature_values)
            
            # Predict
            prediction = model.predict(scaled_features)[0]
            probabilities = model.predict_proba(scaled_features)[0]
            predicted_label = label_mapping[prediction]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📥 Input Features")
                st.json({name: f"{value:.4f}" for name, value in zip(feature_names[:10], sample[feature_names].values[:10])})
                st.caption("Showing first 10 features...")
            
            with col2:
                st.subheader("📤 Prediction Result")
                
                # Actual vs Predicted
                if actual_label == predicted_label:
                    st.success(f"✅ **Correct Prediction!**")
                else:
                    st.error(f"❌ **Incorrect Prediction**")
                
                st.markdown(f"**Actual:** {actual_label}")
                st.markdown(f"**Predicted:** {predicted_label}")
                
                # Confidence
                confidence = probabilities[prediction] * 100
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
                # Progress bar for confidence
                st.progress(probabilities[prediction])
            
            # Probability distribution
            st.subheader("📊 Prediction Probabilities")
            
            prob_df = pd.DataFrame({
                'Attack Type': [label_mapping[i] for i in range(len(probabilities))],
                'Probability': probabilities
            }).sort_values('Probability', ascending=True)
            
            fig = px.bar(
                prob_df,
                x='Probability',
                y='Attack Type',
                orientation='h',
                color='Probability',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ Model or data not loaded.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: gray;">🛡️ NIDS - Network Intrusion Detection System | '
        'Built with Scikit-learn & Streamlit</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
