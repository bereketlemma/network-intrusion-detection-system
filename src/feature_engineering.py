"""
Feature Engineering Module for NIDS
====================================
This module handles all data preprocessing steps:
1. Data cleaning (remove duplicates, handle infinities)
2. Feature scaling (StandardScaler - important for ML!)
3. Label encoding (convert text labels to numbers)
4. Train/test split
5. Handle class imbalance (undersampling)

Why Feature Engineering Matters:
- ML algorithms work better with scaled data
- Class imbalance can bias models toward majority class
- Proper preprocessing improves model accuracy significantly
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple

# Optional: For handling class imbalance
try:
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️  imbalanced-learn not installed. Run: pip install imbalanced-learn")


# ============================================================
# CONFIGURATION
# ============================================================
LABEL_COLUMN = "Attack Type"
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% test, 80% train


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling common issues.
    
    Steps:
    1. Remove duplicate rows
    2. Replace infinite values
    3. Handle NaN values
    """
    print("\n" + "=" * 60)
    print("🧹 DATA CLEANING")
    print("=" * 60)
    
    initial_rows = len(df)
    
    # Step 1: Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"   ✓ Removed {duplicates_removed:,} duplicate rows")
    
    # Step 2: Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"   ✓ Found {len(numeric_cols)} numeric columns")
    
    # Step 3: Replace infinite values with column max/min
    inf_count = 0
    for col in numeric_cols:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            inf_count += inf_mask.sum()
            col_max = df.loc[~inf_mask, col].max()
            col_min = df.loc[~inf_mask, col].min()
            df.loc[df[col] == np.inf, col] = col_max
            df.loc[df[col] == -np.inf, col] = col_min
    print(f"   ✓ Replaced {inf_count:,} infinite values")
    
    # Step 4: Handle NaN values
    nan_count = df[numeric_cols].isna().sum().sum()
    if nan_count > 0:
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        print(f"   ✓ Filled {nan_count:,} NaN values")
    else:
        print(f"   ✓ No NaN values found")
    
    print(f"\n   📊 Final: {len(df):,} rows")
    return df


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, dict]:
    """
    Convert text labels to numbers.
    
    Example:
        'Normal Traffic' -> 0
        'DoS' -> 1
        'DDoS' -> 2
        etc.
    """
    print("\n" + "=" * 60)
    print("🏷️  LABEL ENCODING")
    print("=" * 60)
    
    label_encoder = LabelEncoder()
    df['Label_Encoded'] = label_encoder.fit_transform(df[LABEL_COLUMN])
    
    # Create mapping dictionary
    label_mapping = dict(zip(
        label_encoder.transform(label_encoder.classes_),
        label_encoder.classes_
    ))
    
    print(f"   Label mapping:")
    for code, name in sorted(label_mapping.items()):
        count = (df['Label_Encoded'] == code).sum()
        print(f"      {code} → {name} ({count:,} samples)")
    
    return df, label_encoder, label_mapping


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) from target (y).
    """
    print("\n" + "=" * 60)
    print("📦 PREPARING FEATURES")
    print("=" * 60)
    
    # Columns to exclude from features
    exclude_cols = [LABEL_COLUMN, 'Label_Encoded']
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['Label_Encoded'].copy()
    
    print(f"   ✓ Features (X): {X.shape[1]} columns")
    print(f"   ✓ Target (y): {len(y):,} samples")
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple:
    """
    Split data into training and testing sets.
    Uses stratified sampling to maintain class proportions.
    """
    print("\n" + "=" * 60)
    print("✂️  TRAIN/TEST SPLIT")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"   ✓ Training set: {len(X_train):,} samples ({100-TEST_SIZE*100:.0f}%)")
    print(f"   ✓ Testing set: {len(X_test):,} samples ({TEST_SIZE*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Why scale?
    - ML algorithms perform better with normalized data
    - StandardScaler: mean=0, std=1
    
    IMPORTANT: Fit scaler on training data ONLY!
    """
    print("\n" + "=" * 60)
    print("⚖️  FEATURE SCALING")
    print("=" * 60)
    
    scaler = StandardScaler()
    
    # Fit on training, transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   ✓ Scaler fitted on training data")
    print(f"   ✓ Training shape: {X_train_scaled.shape}")
    print(f"   ✓ Testing shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def handle_imbalance(X_train: np.ndarray, y_train: np.ndarray, 
                     method: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance in training data.
    
    Methods:
    - 'undersample': Reduce majority class (faster)
    - 'none': Skip resampling
    
    Why handle imbalance?
    - 83% Normal Traffic would bias model to always predict "Normal"
    - Undersampling creates balanced classes for better learning
    """
    print("\n" + "=" * 60)
    print("🎯 HANDLING CLASS IMBALANCE")
    print("=" * 60)
    
    print(f"   Method: {method}")
    print(f"\n   Before resampling:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"      Class {label}: {count:,} samples")
    
    if method == 'none':
        print("   → Skipping resampling")
        return X_train, y_train
    
    if not IMBLEARN_AVAILABLE:
        print("   ⚠️  imbalanced-learn not installed, skipping")
        return X_train, y_train
    
    if method == 'undersample':
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    else:
        return X_train, y_train
    
    print(f"\n   After resampling:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"      Class {label}: {count:,} samples")
    
    print(f"\n   📊 Total: {len(y_train):,} → {len(y_resampled):,}")
    
    return X_resampled, y_resampled


def save_objects(scaler: StandardScaler, label_encoder: LabelEncoder,
                 label_mapping: dict, feature_names: list,
                 save_dir: str = "models") -> None:
    """
    Save preprocessing objects for later use.
    """
    print("\n" + "=" * 60)
    print("💾 SAVING PREPROCESSING OBJECTS")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.joblib'))
    joblib.dump(label_mapping, os.path.join(save_dir, 'label_mapping.joblib'))
    joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.joblib'))
    
    print(f"   ✓ Saved to {save_dir}/")
    print(f"      - scaler.joblib")
    print(f"      - label_encoder.joblib")
    print(f"      - label_mapping.joblib")
    print(f"      - feature_names.joblib")


def run_preprocessing_pipeline(df: pd.DataFrame, 
                                balance_method: str = 'undersample',
                                save: bool = True) -> dict:
    """
    Run the complete preprocessing pipeline.
    
    Steps:
    1. Clean data
    2. Encode labels
    3. Prepare features
    4. Split train/test
    5. Scale features
    6. Handle imbalance
    7. Save objects
    """
    print("\n" + "=" * 60)
    print("🚀 PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"   Input: {len(df):,} rows × {len(df.columns)} columns")
    
    # Step 1: Clean
    df_clean = clean_data(df)
    
    # Step 2: Encode labels
    df_encoded, label_encoder, label_mapping = encode_labels(df_clean)
    
    # Step 3: Prepare features
    X, y = prepare_features(df_encoded)
    feature_names = X.columns.tolist()
    
    # Step 4: Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 6: Handle imbalance (training data only!)
    X_train_balanced, y_train_balanced = handle_imbalance(
        X_train_scaled, y_train.values, method=balance_method
    )
    
    # Step 7: Save
    if save:
        save_objects(scaler, label_encoder, label_mapping, feature_names)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"   Training samples: {len(y_train_balanced):,}")
    print(f"   Testing samples: {len(y_test):,}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Classes: {len(label_mapping)}")
    
    return {
        'X_train': X_train_balanced,
        'X_test': X_test_scaled,
        'y_train': y_train_balanced,
        'y_test': y_test.values,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'label_mapping': label_mapping,
        'feature_names': feature_names
    }


# ============================================================
# MAIN - Test the module
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 FEATURE ENGINEERING - TEST RUN")
    print("=" * 60)
    
    from data_loader import load_data
    
    data_file = "data/cicids2017_cleaned.csv"
    
    try:
        # Load sample for testing
        print("\n📂 Loading sample data (100K rows for quick test)...")
        df = load_data(data_file, sample_size=100000)
        
        # Run pipeline
        results = run_preprocessing_pipeline(df, balance_method='undersample')
        
        print("\n" + "=" * 60)
        print("📊 RESULTS SUMMARY")
        print("=" * 60)
        print(f"   X_train shape: {results['X_train'].shape}")
        print(f"   X_test shape: {results['X_test'].shape}")
        print(f"   y_train shape: {results['y_train'].shape}")
        print(f"   y_test shape: {results['y_test'].shape}")
        
        print("\n✅ Feature engineering test complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
