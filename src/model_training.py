"""
Model Training Module for NIDS
==============================
This module trains machine learning models for intrusion detection.

Models available:
1. Random Forest (recommended) - Fast, accurate, interpretable
2. Decision Tree - Simple baseline
3. Logistic Regression - Linear baseline

Why Random Forest?
- Excellent for tabular data with many features
- Handles non-linear relationships
- Provides feature importance rankings
- Resistant to overfitting
- Fast training and prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
import os
import time
from typing import Tuple, Optional


# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
MODELS_DIR = "models"


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = 100,
                        max_depth: Optional[int] = None,
                        n_jobs: int = -1) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features (scaled)
    y_train : np.ndarray
        Training labels
    n_estimators : int
        Number of trees in the forest (default: 100)
    max_depth : int, optional
        Maximum depth of trees (None = unlimited)
    n_jobs : int
        Number of CPU cores to use (-1 = all cores)
    
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    print("\n" + "=" * 60)
    print("🌲 TRAINING RANDOM FOREST")
    print("=" * 60)
    
    print(f"   Parameters:")
    print(f"      - n_estimators: {n_estimators}")
    print(f"      - max_depth: {max_depth}")
    print(f"      - n_jobs: {n_jobs} (CPU cores)")
    print(f"   Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        class_weight='balanced'  # Handle any remaining imbalance
    )
    
    print(f"\n   Training in progress...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"   ✓ Training complete in {training_time:.2f} seconds")
    
    # Training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f"   ✓ Training accuracy: {train_accuracy * 100:.2f}%")
    
    return model


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                        max_depth: Optional[int] = 20) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier (simpler baseline).
    """
    print("\n" + "=" * 60)
    print("🌳 TRAINING DECISION TREE")
    print("=" * 60)
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    
    print(f"   Training data: {X_train.shape[0]:,} samples")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"   ✓ Training complete in {training_time:.2f} seconds")
    print(f"   ✓ Training accuracy: {model.score(X_train, y_train) * 100:.2f}%")
    
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   label_mapping: dict) -> dict:
    """
    Evaluate model performance on test data.
    
    Metrics calculated:
    - Accuracy: Overall correctness
    - Precision: Of predicted attacks, how many were real?
    - Recall: Of real attacks, how many did we catch?
    - F1-Score: Balance of precision and recall
    
    Parameters:
    -----------
    model : trained classifier
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        True labels
    label_mapping : dict
        Mapping of label numbers to names
    
    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION")
    print("=" * 60)
    
    # Make predictions
    print("   Making predictions...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    print(f"   ✓ Predicted {len(y_test):,} samples in {pred_time:.2f} seconds")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Display results
    print(f"\n   📈 OVERALL METRICS:")
    print(f"      Accuracy:  {accuracy * 100:.2f}%")
    print(f"      Precision: {precision * 100:.2f}%")
    print(f"      Recall:    {recall * 100:.2f}%")
    print(f"      F1-Score:  {f1 * 100:.2f}%")
    
    # Per-class metrics
    print(f"\n   📋 PER-CLASS PERFORMANCE:")
    
    # Get class names in order
    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics


def get_feature_importance(model, feature_names: list, top_n: int = 15) -> pd.DataFrame:
    """
    Get feature importance rankings from the model.
    
    Shows which network features are most important for detecting attacks.
    
    Parameters:
    -----------
    model : trained model (must have feature_importances_ attribute)
    feature_names : list
        Names of all features
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    pd.DataFrame
        Feature importance rankings
    """
    print("\n" + "=" * 60)
    print("🔍 FEATURE IMPORTANCE")
    print("=" * 60)
    
    if not hasattr(model, 'feature_importances_'):
        print("   ⚠️  Model doesn't support feature importance")
        return None
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n   Top {top_n} Most Important Features:")
    print(f"   {'Rank':<6}{'Feature':<35}{'Importance':<12}")
    print("   " + "-" * 53)
    
    for i, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
        print(f"   {i:<6}{row['Feature']:<35}{row['Importance']:.4f}")
    
    return importance_df


def save_model(model, model_name: str = "random_forest_model.joblib",
               save_dir: str = MODELS_DIR) -> str:
    """
    Save trained model to disk.
    """
    print("\n" + "=" * 60)
    print("💾 SAVING MODEL")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)
    
    joblib.dump(model, model_path)
    
    # Get file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    print(f"   ✓ Model saved: {model_path}")
    print(f"   ✓ File size: {file_size:.2f} MB")
    
    return model_path


def load_model(model_path: str):
    """
    Load a trained model from disk.
    """
    print(f"   Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"   ✓ Model loaded successfully")
    return model


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       label_mapping: dict, feature_names: list,
                       model_type: str = 'random_forest',
                       save: bool = True) -> Tuple:
    """
    Complete training and evaluation pipeline.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    label_mapping : dict mapping label numbers to names
    feature_names : list of feature column names
    model_type : 'random_forest' or 'decision_tree'
    save : Whether to save the model
    
    Returns:
    --------
    Tuple containing (model, metrics, feature_importance)
    """
    print("\n" + "=" * 60)
    print("🚀 TRAINING & EVALUATION PIPELINE")
    print("=" * 60)
    
    # Train model
    if model_type == 'random_forest':
        model = train_random_forest(X_train, y_train)
        model_filename = "random_forest_model.joblib"
    elif model_type == 'decision_tree':
        model = train_decision_tree(X_train, y_train)
        model_filename = "decision_tree_model.joblib"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, label_mapping)
    
    # Feature importance
    importance_df = get_feature_importance(model, feature_names)
    
    # Save model
    if save:
        save_model(model, model_filename)
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Model: {model_type}")
    print(f"   Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"   Test F1-Score: {metrics['f1_score'] * 100:.2f}%")
    
    return model, metrics, importance_df


# ============================================================
# MAIN - Test the module
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 MODEL TRAINING - TEST RUN")
    print("=" * 60)
    
    from data_loader import load_data
    from feature_engineering import run_preprocessing_pipeline
    
    data_file = "data/cicids2017_cleaned.csv"
    
    try:
        # Load data (use larger sample for better training)
        print("\n📂 Loading data...")
        df = load_data(data_file, sample_size=None)  # 200K sample
        
        # Preprocess
        print("\n⚙️ Preprocessing...")
        data = run_preprocessing_pipeline(df, balance_method='undersample')
        
        # Train and evaluate
        model, metrics, importance = train_and_evaluate(
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            label_mapping=data['label_mapping'],
            feature_names=data['feature_names'],
            model_type='random_forest'
        )
        
        print("\n" + "=" * 60)
        print("🎉 MODEL TRAINING TEST COMPLETE!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
