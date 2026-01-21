"""
NIDS Interactive Dashboard - Improved Version
==============================================
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
# IMPROVED CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5, #00C851);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    
    /* Metric cards with better visibility */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #0f3460;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ff88;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Attack alert styling - RED */
    .attack-alert {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 12px 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 5px solid #ff0000;
        box-shadow: 0 2px 4px rgba(255, 0, 0, 0.3);
    }
    
    .attack-alert b {
        color: #ffff00;
    }
    
    .attack-alert small {
        color: #ffcccc;
        font-size: 0.85rem;
    }
    
    /* Normal traffic styling - GREEN */
    .normal-traffic {
        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
        color: white;
        padding: 12px 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 5px solid #00ff00;
        box-shadow: 0 2px 4px rgba(0, 200, 81, 0.3);
    }
    
    .normal-traffic small {
        color: #ccffcc;
        font-size: 0.85rem;
    }
    
    /* Threat level badges */
    .threat-low {
        background: linear-gradient(135deg, #00C851, #007E33);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 200, 81, 0.4);
    }
    
    .threat-medium {
        background: linear-gradient(135deg, #ffbb33, #ff8800);
        color: #000;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(255, 187, 51, 0.4);
    }
    
    .threat-high {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(255, 68, 68, 0.4);
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
    }
    
    /* Info boxes */
    .info-box {
        background: #1e3a5f;
        border: 1px solid #2e5a8f;
        border-radius: 10px;
        padding: 15px;
        color: #fff;
        margin: 10px 0;
    }
    
    .info-box-title {
        color: #00C851;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    /* Performance metric boxes */
    .perf-metric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 2px solid;
        margin: 10px 0;
    }
    
    .perf-metric.accuracy { border-color: #00C851; }
    .perf-metric.precision { border-color: #1E88E5; }
    .perf-metric.recall { border-color: #ffbb33; }
    .perf-metric.f1 { border-color: #aa66cc; }
    
    .perf-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .perf-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
    }
    
    /* Sidebar styling */
    .sidebar-status {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .sidebar-status.online {
        background-color: rgba(0, 200, 81, 0.2);
        border: 1px solid #00C851;
        color: #00C851;
    }
    
    .sidebar-status.offline {
        background-color: rgba(255, 68, 68, 0.2);
        border: 1px solid #ff4444;
        color: #ff4444;
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
def load_sample_data(n_samples=50000):
    """Load a RANDOM sample of the dataset for visualization."""
    data_file = "data/cicids2017_cleaned.csv"
    try:
        # Load full dataset and take random sample to get all attack types
        df = pd.read_csv(data_file, low_memory=False)
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42)
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
    
    # Weighted probabilities
    weights = [0.75, 0.10, 0.06, 0.04, 0.02, 0.02, 0.01]
    
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

def render_metric_card(icon, value, label, color="#00ff88"):
    """Render a custom metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def render_threat_level(level):
    """Render threat level badge."""
    if level < 3:
        return '<div class="threat-low">🟢 LOW THREAT</div>'
    elif level < 7:
        return '<div class="threat-medium">🟡 MEDIUM THREAT</div>'
    else:
        return '<div class="threat-high">🔴 HIGH THREAT</div>'

# ============================================================
# MAIN DASHBOARD
# ============================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Network Intrusion Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time AI-powered network security monitoring</p>', 
                unsafe_allow_html=True)
    
    # Load model
    model, scaler, label_mapping, feature_names, model_loaded = load_model_and_preprocessors()
    
    # Load sample data
    df, data_loaded = load_sample_data()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    page = st.sidebar.radio(
        "Navigate",
        ["📊 Dashboard", "🔴 Live Monitor", "📈 Model Performance", "🔍 Feature Analysis", "🎯 Test Prediction"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 System Status")
    
    if model_loaded:
        st.sidebar.markdown('<div class="sidebar-status online">✅ Model: ONLINE</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="sidebar-status offline">❌ Model: OFFLINE</div>', unsafe_allow_html=True)
        st.sidebar.caption("Run `python src/model_training.py`")
    
    if data_loaded:
        st.sidebar.markdown('<div class="sidebar-status online">✅ Data: LOADED</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="sidebar-status offline">❌ Data: NOT FOUND</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.caption("NIDS v1.0 - Built with Scikit-learn, Streamlit & Plotly")
    
    # ============================================================
    # PAGE: DASHBOARD
    # ============================================================
    if page == "📊 Dashboard":
        st.markdown("## 📊 Security Overview Dashboard")
        
        if data_loaded:
            # Calculate metrics
            total_packets = len(df)
            attack_count = len(df[df['Attack Type'] != 'Normal Traffic'])
            normal_count = len(df[df['Attack Type'] == 'Normal Traffic'])
            attack_rate = (attack_count / total_packets) * 100
            
            # Top metrics with custom cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(render_metric_card("📦", f"{total_packets:,}", "Total Packets", "#00C851"), unsafe_allow_html=True)
            with col2:
                st.markdown(render_metric_card("✅", f"{normal_count:,}", "Normal Traffic", "#1E88E5"), unsafe_allow_html=True)
            with col3:
                st.markdown(render_metric_card("🚨", f"{attack_count:,}", "Attacks Detected", "#ff4444"), unsafe_allow_html=True)
            with col4:
                st.markdown(render_metric_card("⚠️", f"{attack_rate:.2f}%", "Attack Rate", "#ffbb33"), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Charts row
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Attack Distribution")
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
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#fff',
                    legend=dict(font=dict(color='#fff'))
                )
                fig.update_traces(textinfo='percent+label', textfont_size=12)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Attack Type Counts")
                
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
                    },
                    text=attack_counts.values
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Attack Type",
                    yaxis_title="Count",
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#fff',
                    xaxis=dict(tickfont=dict(color='#fff')),
                    yaxis=dict(tickfont=dict(color='#fff'), gridcolor='rgba(255,255,255,0.1)')
                )
                fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            # Attack details table
            st.markdown("### 📋 Attack Summary Table")
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
        st.markdown("## 🔴 Live Network Traffic Monitor")
        st.caption("Simulated real-time network packet analysis")
        
        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)
        with col2:
            refresh_rate = st.slider("Refresh Rate (sec)", 1, 10, 3)
        with col3:
            if st.button("🔄 Refresh Now", type="primary"):
                st.rerun()
        
        # Simulate traffic
        traffic_df = simulate_network_traffic(25)
        
        # Count attacks
        attacks = traffic_df[traffic_df['attack_type'] != 'Normal Traffic']
        attack_count = len(attacks)
        
        st.markdown("---")
        
        # Live stats with custom cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(render_metric_card("📡", str(len(traffic_df)), "Packets/sec", "#1E88E5"), unsafe_allow_html=True)
        with col2:
            color = "#ff4444" if attack_count > 0 else "#00C851"
            st.markdown(render_metric_card("🚨", str(attack_count), "Threats", color), unsafe_allow_html=True)
        with col3:
            st.markdown(render_metric_card("🌐", str(traffic_df['src_ip'].nunique()), "Unique IPs", "#aa66cc"), unsafe_allow_html=True)
        with col4:
            st.markdown(render_threat_level(attack_count), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Live traffic feed
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📜 Live Traffic Feed")
            
            for _, packet in traffic_df.iterrows():
                is_attack = packet['attack_type'] != 'Normal Traffic'
                
                if is_attack:
                    st.markdown(f"""
                    <div class="attack-alert">
                        🚨 <b>ALERT: {packet['attack_type']} detected!</b><br>
                        <small>🕐 {packet['timestamp'].strftime('%H:%M:%S')} | 
                        📤 {packet['src_ip']}:{packet['src_port']} → 
                        📥 {packet['dst_ip']}:{packet['dst_port']} | 
                        📡 {packet['protocol']} | 📊 {packet['bytes']:,} bytes</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="normal-traffic">
                        ✅ <b>Normal Traffic</b><br>
                        <small>🕐 {packet['timestamp'].strftime('%H:%M:%S')} | 
                        📤 {packet['src_ip']}:{packet['src_port']} → 
                        📥 {packet['dst_ip']}:{packet['dst_port']} | 
                        📡 {packet['protocol']} | 📊 {packet['bytes']:,} bytes</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎯 Traffic Breakdown")
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
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#fff',
                legend=dict(font=dict(color='#fff', size=10))
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Attack breakdown
            st.markdown("#### 📊 Counts")
            for attack_type, count in attack_summary.items():
                color = get_attack_color(attack_type)
                pct = count / len(traffic_df) * 100
                st.markdown(f"<span style='color:{color};font-weight:bold;'>{attack_type}</span>: {count} ({pct:.1f}%)", unsafe_allow_html=True)
        
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
    
    # ============================================================
    # PAGE: MODEL PERFORMANCE
    # ============================================================
    elif page == "📈 Model Performance":
        st.markdown("## 📈 Model Performance Metrics")
        
        if model_loaded:
            # Model info
            st.markdown("### 🤖 Model Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="info-box">
                    <div class="info-box-title">Model Type</div>
                    <div style="font-size: 1.3rem;">🌲 Random Forest</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="info-box">
                    <div class="info-box-title">Number of Trees</div>
                    <div style="font-size: 1.3rem;">🌳 {model.n_estimators}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="info-box">
                    <div class="info-box-title">Features Used</div>
                    <div style="font-size: 1.3rem;">📊 {len(feature_names)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Performance metrics
            st.markdown("### 📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="perf-metric accuracy">
                    <div class="perf-label">🎯 Accuracy</div>
                    <div class="perf-value" style="color: #00C851;">93.13%</div>
                    <div style="color: #888;">Target: 95%</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="perf-metric precision">
                    <div class="perf-label">📌 Precision</div>
                    <div class="perf-value" style="color: #1E88E5;">97.56%</div>
                    <div style="color: #00C851;">+2.56% ▲</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="perf-metric recall">
                    <div class="perf-label">🔍 Recall</div>
                    <div class="perf-value" style="color: #ffbb33;">93.13%</div>
                    <div style="color: #ff4444;">-1.87% ▼</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown("""
                <div class="perf-metric f1">
                    <div class="perf-label">⚖️ F1-Score</div>
                    <div class="perf-value" style="color: #aa66cc;">95.06%</div>
                    <div style="color: #00C851;">✅ Target Met</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Per-class performance
            st.markdown("### 📋 Per-Class Performance")
            
            performance_data = {
                'Attack Type': ['Bots', 'Brute Force', 'DDoS', 'DoS', 'Normal Traffic', 'Port Scanning', 'Web Attacks'],
                'Precision': [0.04, 0.48, 0.99, 0.79, 1.00, 0.95, 0.04],
                'Recall': [1.00, 0.97, 1.00, 0.95, 0.92, 0.99, 1.00],
                'F1-Score': [0.08, 0.64, 0.99, 0.86, 0.96, 0.97, 0.08],
                'Support': [33, 152, 2019, 3084, 33224, 1451, 37]
            }
            perf_df = pd.DataFrame(performance_data)
            
            # Bar chart
            fig = go.Figure()
            
            colors = {'Precision': '#1E88E5', 'Recall': '#00C851', 'F1-Score': '#aa66cc'}
            
            for metric in ['Precision', 'Recall', 'F1-Score']:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=perf_df['Attack Type'],
                    y=perf_df[metric],
                    text=[f"{v:.0%}" for v in perf_df[metric]],
                    textposition='auto',
                    marker_color=colors[metric]
                ))
            
            fig.update_layout(
                barmode='group',
                height=450,
                xaxis_title="Attack Type",
                yaxis_title="Score",
                yaxis_range=[0, 1.15],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#fff',
                legend=dict(font=dict(color='#fff')),
                xaxis=dict(tickfont=dict(color='#fff')),
                yaxis=dict(tickfont=dict(color='#fff'), gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
            
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
        st.markdown("## 🔍 Feature Importance Analysis")
        st.caption("Which network features are most important for detecting attacks?")
        
        if model_loaded:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            top_features = importance_df.tail(15)
            
            fig = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis',
                text=[f"{v:.3f}" for v in top_features['Importance']]
            )
            fig.update_layout(
                height=550,
                title="Top 15 Most Important Features",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#fff',
                xaxis=dict(tickfont=dict(color='#fff'), gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(tickfont=dict(color='#fff'))
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📖 Feature Descriptions")
            
            feature_descriptions = {
                'Destination Port': '🔌 Target port number - key indicator of attack type',
                'Init_Win_bytes_backward': '📊 TCP window size in backward direction',
                'Packet Length Mean': '📏 Average size of packets in the flow',
                'Max Packet Length': '📐 Largest packet in the flow',
                'Subflow Fwd Bytes': '➡️ Total bytes sent in forward direction',
                'Total Length of Fwd Packets': '📦 Sum of all forward packet sizes',
                'Average Packet Size': '📊 Mean packet size across the flow',
                'Fwd Packet Length Mean': '➡️ Average forward packet size',
                'Flow IAT Max': '⏱️ Maximum inter-arrival time between packets',
                'Flow Packets/s': '🚀 Packet rate per second'
            }
            
            col1, col2 = st.columns(2)
            features_list = list(top_features.tail(10)['Feature'].values[::-1])
            
            for i, feature in enumerate(features_list):
                col = col1 if i < 5 else col2
                if feature in feature_descriptions:
                    col.markdown(f"""
                    <div class="info-box">
                        <div class="info-box-title">{feature}</div>
                        <div>{feature_descriptions[feature]}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model not loaded. Run training first.")
    
    # ============================================================
    # PAGE: TEST PREDICTION
    # ============================================================
    elif page == "🎯 Test Prediction":
        st.markdown("## 🎯 Test Attack Prediction")
        st.caption("Select a sample from the dataset to see the model's prediction")
        
        if model_loaded and data_loaded:
            col1, col2 = st.columns([3, 1])
            with col1:
                sample_idx = st.slider("Select Sample Index", 0, len(df)-1, 100)
            with col2:
                if st.button("🎲 Random Sample", type="primary"):
                    sample_idx = random.randint(0, len(df)-1)
            
            sample = df.iloc[sample_idx]
            actual_label = sample['Attack Type']
            
            feature_values = sample[feature_names].values.reshape(1, -1)
            scaled_features = scaler.transform(feature_values)
            
            prediction = model.predict(scaled_features)[0]
            probabilities = model.predict_proba(scaled_features)[0]
            predicted_label = label_mapping[prediction]
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📥 Input Features")
                
                feature_data = {name: f"{value:.4f}" for name, value in zip(feature_names[:12], sample[feature_names].values[:12])}
                
                for name, value in feature_data.items():
                    st.markdown(f"""
                    <div style="background: #1a1a2e; padding: 8px 12px; margin: 3px 0; border-radius: 5px; border-left: 3px solid #1E88E5;">
                        <span style="color: #888;">{name}:</span> <span style="color: #00C851; font-weight: bold;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.caption("Showing first 12 of 52 features...")
            
            with col2:
                st.markdown("### 📤 Prediction Result")
                
                is_correct = actual_label == predicted_label
                
                if is_correct:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #00C851, #007E33); padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <div style="font-size: 2rem;">✅</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: white;">Correct Prediction!</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #ff4444, #cc0000); padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <div style="font-size: 2rem;">❌</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: white;">Incorrect Prediction</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                actual_color = "#00C851" if actual_label == "Normal Traffic" else "#ff4444"
                pred_color = "#00C851" if predicted_label == "Normal Traffic" else "#ff4444"
                
                st.markdown(f"""
                <div class="info-box">
                    <div><span style="color: #888;">Actual:</span> <span style="color: {actual_color}; font-weight: bold; font-size: 1.1rem;">{actual_label}</span></div>
                    <div style="margin-top: 10px;"><span style="color: #888;">Predicted:</span> <span style="color: {pred_color}; font-weight: bold; font-size: 1.1rem;">{predicted_label}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                confidence = probabilities[prediction] * 100
                conf_color = "#00C851" if confidence > 70 else ("#ffbb33" if confidence > 40 else "#ff4444")
                
                st.markdown(f"""
                <div class="info-box" style="margin-top: 15px;">
                    <div class="info-box-title">Confidence Level</div>
                    <div style="font-size: 2rem; font-weight: bold; color: {conf_color};">{confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(probabilities[prediction])
            
            st.markdown("### 📊 Prediction Probabilities")
            
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
                color_continuous_scale='RdYlGn',
                text=[f"{p:.1%}" for p in prob_df['Probability']]
            )
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#fff',
                xaxis=dict(tickfont=dict(color='#fff'), gridcolor='rgba(255,255,255,0.1)', tickformat='.0%'),
                yaxis=dict(tickfont=dict(color='#fff'))
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ Model or data not loaded.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">🛡️ NIDS v1.0 - Network Intrusion Detection System | '
        'Built with Scikit-learn, Streamlit & Plotly | © 2026</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()