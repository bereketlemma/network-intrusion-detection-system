"""
Data Loader Module for NIDS
===========================
This module handles loading the CICIDS2017 dataset and provides
functions for initial data exploration.

What this file does:
1. Loads CSV data (handles large files efficiently)
2. Displays dataset information (shape, columns, types)
3. Shows attack distribution (how many of each attack type)
4. Handles memory optimization for large datasets
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional


def load_data(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load the CICIDS2017 dataset from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    sample_size : int, optional
        If provided, randomly sample this many rows (useful for testing)
    
    Returns:
    --------
    pd.DataFrame
        The loaded dataset
    """
    print(f"📂 Loading data from: {file_path}")
    print("-" * 50)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path, low_memory=False)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Total rows: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"📊 Sampling {sample_size:,} rows...")
        df = df.sample(n=sample_size, random_state=42)
        print(f"   Sampled rows: {len(df):,}")
    
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """
    Perform initial exploration of the dataset.
    """
    print("\n" + "=" * 60)
    print("📊 DATASET EXPLORATION")
    print("=" * 60)
    
    # Basic info
    print(f"\n📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"💾 Memory Usage: {memory_mb:.2f} MB")
    
    # Column types
    print(f"\n📋 Column Types:")
    print(f"   - Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   - Object columns: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    print(f"\n❓ Missing Values:")
    if len(missing_cols) > 0:
        print(f"   - {len(missing_cols)} columns have missing values")
        for col in missing_cols.index[:5]:
            print(f"      • {col}: {missing_cols[col]:,} missing")
    else:
        print("   - No missing values! ✅")
    
    # Infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    print(f"\n♾️ Infinite Values:")
    if inf_counts:
        print(f"   - {len(inf_counts)} columns have infinite values")
        for col, count in list(inf_counts.items())[:5]:
            print(f"      • {col}: {count:,} infinite")
    else:
        print("   - No infinite values! ✅")
    
    return {
        'shape': df.shape,
        'memory_mb': memory_mb,
        'missing_columns': missing_cols.to_dict() if len(missing_cols) > 0 else {},
        'infinite_columns': inf_counts,
        'numeric_columns': list(numeric_cols),
        'object_columns': list(df.select_dtypes(include=['object']).columns)
    }


def show_attack_distribution(df: pd.DataFrame, label_column: str = 'Attack Type') -> pd.DataFrame:
    """
    Display the distribution of attack types in the dataset.
    """
    print("\n" + "=" * 60)
    print("🎯 ATTACK DISTRIBUTION")
    print("=" * 60)
    
    # Check if label column exists
    if label_column not in df.columns:
        possible_labels = ['Label', 'label', 'LABEL', 'Attack', 'attack', 'Attack Type']
        for col in possible_labels:
            if col in df.columns:
                label_column = col
                break
        else:
            print(f"❌ Could not find label column. Available columns:")
            print(df.columns.tolist())
            return None
    
    print(f"   Using label column: '{label_column}'")
    
    # Calculate distribution
    attack_counts = df[label_column].value_counts()
    attack_percentages = (attack_counts / len(df) * 100).round(2)
    
    # Create summary DataFrame
    distribution = pd.DataFrame({
        'Attack Type': attack_counts.index,
        'Count': attack_counts.values,
        'Percentage (%)': attack_percentages.values
    })
    
    print(f"\n   Total unique labels: {len(attack_counts)}")
    print("\n" + distribution.to_string(index=False))
    
    # Highlight imbalance
    print("\n⚠️  Class Imbalance Note:")
    if 'Normal Traffic' in attack_counts.index:
        benign_pct = attack_percentages['Normal Traffic']
        if benign_pct > 80:
            print(f"   Dataset is highly imbalanced ({benign_pct:.1f}% normal traffic)")
            print("   We'll handle this in feature engineering!")
    
    return distribution


def show_columns(df: pd.DataFrame) -> None:
    """
    Display all column names.
    """
    print("\n" + "=" * 60)
    print("📝 ALL COLUMNS")
    print("=" * 60)
    
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2}. {col}")


# ============================================================
# MAIN - Run this file directly to test
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 NIDS DATA LOADER - TEST RUN")
    print("=" * 60)
    
    # Your data file path
    data_file = "data/cicids2017_cleaned.csv"
    
    try:
        # Load data (sample 100K rows for quick testing)
        df = load_data(data_file, sample_size=100000)
        
        # Explore data
        exploration = explore_data(df)
        
        # Show attack distribution
        distribution = show_attack_distribution(df)
        
        # Show all columns
        show_columns(df)
        
        print("\n" + "=" * 60)
        print("✅ DATA LOADER TEST COMPLETE!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n💡 Make sure your CSV file is in the 'data' folder")
