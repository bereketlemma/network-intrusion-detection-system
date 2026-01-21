"""
Full Dataset Training Script for NIDS
=====================================
This script trains the model on the COMPLETE 2.5M+ dataset
for maximum accuracy on all attack types.

Expected Results:
- Better accuracy on rare attacks (Bots, Web Attacks, Brute Force)
- Training time: 5-15 minutes (depending on your CPU)
- RAM usage: 4-8 GB

Run with:
    python train_full_model.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os
import time
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = "data/cicids2017_cleaned.csv"
MODELS_DIR = "models"
LABEL_COLUMN = "Attack Type"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model parameters - optimized for full dataset
N_ESTIMATORS = 200      # More trees for better accuracy
MAX_DEPTH = 30          # Limit depth to prevent overfitting
MIN_SAMPLES_SPLIT = 10  # Minimum samples to split
MIN_SAMPLES_LEAF = 5    # Minimum samples in leaf
N_JOBS = -1             # Use all CPU cores


def main():
    print("\n" + "=" * 70)
    print("🚀 NIDS - FULL DATASET TRAINING")
    print("=" * 70)
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ============================================================
    # STEP 1: LOAD FULL DATASET
    # ============================================================
    print("\n" + "=" * 70)
    print("📂 STEP 1: Loading Full Dataset")
    print("=" * 70)
    
    start_time = time.time()
    
    print(f"   Loading from: {DATA_FILE}")
    print("   This may take 1-2 minutes...")
    
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    load_time = time.time() - start_time
    print(f"   ✅ Loaded in {load_time:.1f} seconds")
    print(f"   Total rows: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    # ============================================================
    # STEP 2: DATA CLEANING
    # ============================================================
    print("\n" + "=" * 70)
    print("🧹 STEP 2: Data Cleaning")
    print("=" * 70)
    
    initial_rows = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"   Removed {initial_rows - len(df):,} duplicate rows")
    
    # Handle infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            col_max = df.loc[~inf_mask, col].max()
            col_min = df.loc[~inf_mask, col].min()
            df.loc[df[col] == np.inf, col] = col_max
            df.loc[df[col] == -np.inf, col] = col_min
    
    # Handle NaN
    df = df.fillna(df.median(numeric_only=True))
    
    print(f"   ✅ Clean data: {len(df):,} rows")
    
    # ============================================================
    # STEP 3: SHOW ATTACK DISTRIBUTION
    # ============================================================
    print("\n" + "=" * 70)
    print("🎯 STEP 3: Attack Distribution")
    print("=" * 70)
    
    attack_counts = df[LABEL_COLUMN].value_counts()
    print("\n   Attack Type           Count         Percentage")
    print("   " + "-" * 50)
    for attack, count in attack_counts.items():
        pct = count / len(df) * 100
        print(f"   {attack:<20} {count:>10,}    ({pct:>6.2f}%)")
    
    # ============================================================
    # STEP 4: LABEL ENCODING
    # ============================================================
    print("\n" + "=" * 70)
    print("🏷️  STEP 4: Label Encoding")
    print("=" * 70)
    
    label_encoder = LabelEncoder()
    df['Label_Encoded'] = label_encoder.fit_transform(df[LABEL_COLUMN])
    
    label_mapping = dict(zip(
        label_encoder.transform(label_encoder.classes_),
        label_encoder.classes_
    ))
    
    print("   Label mapping:")
    for code, name in sorted(label_mapping.items()):
        print(f"      {code} → {name}")
    
    # ============================================================
    # STEP 5: PREPARE FEATURES
    # ============================================================
    print("\n" + "=" * 70)
    print("📦 STEP 5: Preparing Features")
    print("=" * 70)
    
    exclude_cols = [LABEL_COLUMN, 'Label_Encoded']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['Label_Encoded']
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(y):,}")
    
    # ============================================================
    # STEP 6: TRAIN/TEST SPLIT
    # ============================================================
    print("\n" + "=" * 70)
    print("✂️  STEP 6: Train/Test Split")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"   Training: {len(X_train):,} samples (80%)")
    print(f"   Testing: {len(X_test):,} samples (20%)")
    
    # ============================================================
    # STEP 7: FEATURE SCALING
    # ============================================================
    print("\n" + "=" * 70)
    print("⚖️  STEP 7: Feature Scaling")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   ✅ Scaler fitted and applied")
    
    # ============================================================
    # STEP 8: HANDLE CLASS IMBALANCE
    # ============================================================
    print("\n" + "=" * 70)
    print("🎯 STEP 8: Handling Class Imbalance")
    print("=" * 70)
    
    print("\n   Before resampling:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"      Class {label} ({label_mapping[label]}): {count:,}")
    
    # Use undersampling
    undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_scaled, y_train)
    
    print("\n   After resampling:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"      Class {label} ({label_mapping[label]}): {count:,}")
    
    print(f"\n   Total training samples: {len(y_train):,} → {len(y_train_balanced):,}")
    
    # ============================================================
    # STEP 9: TRAIN MODEL
    # ============================================================
    print("\n" + "=" * 70)
    print("🌲 STEP 9: Training Random Forest (Full Dataset)")
    print("=" * 70)
    
    print(f"\n   Model Parameters:")
    print(f"      - n_estimators: {N_ESTIMATORS}")
    print(f"      - max_depth: {MAX_DEPTH}")
    print(f"      - min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"      - min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"      - n_jobs: {N_JOBS} (all CPU cores)")
    
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        class_weight='balanced',
        verbose=1  # Show progress
    )
    
    print(f"\n   Training in progress...")
    print(f"   (This may take 5-15 minutes)\n")
    
    train_start = time.time()
    model.fit(X_train_balanced, y_train_balanced)
    train_time = time.time() - train_start
    
    print(f"\n   ✅ Training complete in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")
    
    # Training accuracy
    train_accuracy = model.score(X_train_balanced, y_train_balanced)
    print(f"   Training accuracy: {train_accuracy * 100:.2f}%")
    
    # ============================================================
    # STEP 10: EVALUATE MODEL
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 STEP 10: Model Evaluation")
    print("=" * 70)
    
    print(f"\n   Making predictions on {len(X_test):,} test samples...")
    
    pred_start = time.time()
    y_pred = model.predict(X_test_scaled)
    pred_time = time.time() - pred_start
    
    print(f"   Predictions complete in {pred_time:.2f} seconds")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n   📈 OVERALL METRICS:")
    print(f"      Accuracy:  {accuracy * 100:.2f}%")
    print(f"      Precision: {precision * 100:.2f}%")
    print(f"      Recall:    {recall * 100:.2f}%")
    print(f"      F1-Score:  {f1 * 100:.2f}%")
    
    # Classification report
    print(f"\n   📋 CLASSIFICATION REPORT:")
    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(report)
    
    # ============================================================
    # STEP 11: FEATURE IMPORTANCE
    # ============================================================
    print("\n" + "=" * 70)
    print("🔍 STEP 11: Feature Importance")
    print("=" * 70)
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n   Top 15 Most Important Features:")
    print(f"   {'Rank':<6}{'Feature':<35}{'Importance':<12}")
    print("   " + "-" * 53)
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        print(f"   {i:<6}{row['Feature']:<35}{row['Importance']:.4f}")
    
    # ============================================================
    # STEP 12: SAVE EVERYTHING
    # ============================================================
    print("\n" + "=" * 70)
    print("💾 STEP 12: Saving Model and Objects")
    print("=" * 70)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
    joblib.dump(model, model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ Model saved: {model_path} ({model_size:.1f} MB)")
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    print(f"   ✅ Scaler saved")
    
    # Save label encoder
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    print(f"   ✅ Label encoder saved")
    
    # Save label mapping
    joblib.dump(label_mapping, os.path.join(MODELS_DIR, 'label_mapping.joblib'))
    print(f"   ✅ Label mapping saved")
    
    # Save feature names
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_names.joblib'))
    print(f"   ✅ Feature names saved")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🏆 TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"""
   📊 Dataset:
      - Total samples: {len(df):,}
      - Training samples: {len(y_train_balanced):,} (balanced)
      - Testing samples: {len(y_test):,}
      - Features: {len(feature_cols)}

   🤖 Model:
      - Algorithm: Random Forest
      - Trees: {N_ESTIMATORS}
      - Max Depth: {MAX_DEPTH}

   📈 Performance:
      - Accuracy:  {accuracy * 100:.2f}%
      - Precision: {precision * 100:.2f}%
      - Recall:    {recall * 100:.2f}%
      - F1-Score:  {f1 * 100:.2f}%

   ⏱️  Time:
      - Total: {total_time/60:.1f} minutes
      - Training: {train_time/60:.1f} minutes
      - Prediction: {pred_time:.2f} seconds

   📁 Files saved to: {MODELS_DIR}/
    """)
    
    print("=" * 70)
    print("✅ Full dataset training complete!")
    print("   Run 'streamlit run dashboard/app.py' to see results in dashboard")
    print("=" * 70)


if __name__ == "__main__":
    main()
