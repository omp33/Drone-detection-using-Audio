"""
Simplified XGBoost Training Script (No Plotting Issues)
Works perfectly, shows all important metrics
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import json

def train_xgboost_simple(features_csv, output_model='drone_detector_xgb.model'):
    """
    Simple, working XGBoost training
    """
    print("="*80)
    print("XGBOOST DRONE DETECTOR - SIMPLIFIED VERSION".center(80))
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(features_csv)
    feature_cols = [col for col in df.columns if col not in ['label', 'filename', 'class']]
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"✓ Loaded {len(df)} samples with {len(feature_cols)} features")
    print(f"  Drone: {np.sum(y==1)} samples")
    print(f"  Background: {np.sum(y==0)} samples")
    
    # Split data
    print("\n📊 Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1875, random_state=42, stratify=y_temp
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Calculate class imbalance
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos
    
    print(f"\n⚖️  Class balance:")
    print(f"  Background: {n_neg} | Drone: {n_pos}")
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Parameters
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    print("\n🔧 Training XGBoost...")
    print("  Configuration:", json.dumps(params, indent=4))
    
    # Train
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=25
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best validation loss: {model.best_score:.4f}")
    
    # Feature importance
    print("\n" + "="*80)
    print("TOP 15 MOST IMPORTANT FEATURES".center(80))
    print("="*80)
    
    importance = model.get_score(importance_type='gain')
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':<6} {'Feature':<35} {'Importance':>12}")
    print("-"*80)
    for i, (feat, imp) in enumerate(importance_sorted[:15], 1):
        # Try to get readable name
        if feat.startswith('f'):
            try:
                idx = int(feat[1:])
                feat_name = feature_cols[idx]
            except:
                feat_name = feat
        else:
            feat_name = feat
        print(f"{i:<6} {feat_name:<35} {imp:>12.2f}")
    
    # Predictions
    print("\n" + "="*80)
    print("TEST SET EVALUATION".center(80))
    print("="*80)
    
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}  (When says 'drone', correct {precision:.0%} of time)")
    print(f"  Recall:    {recall:.1%}  (Catches {recall:.0%} of all drones)")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Bg    Drone")
    print(f"Actual Bg       {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"       Drone    {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    # Detailed report
    print(f"\n📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Background', 'Drone']))
    
    # Analysis
    fp = cm[0,1]  # False positives
    fn = cm[1,0]  # False negatives
    
    print(f"\n⚠️  Error Analysis:")
    print(f"  False Positives: {fp} (background mistaken as drone)")
    print(f"  False Negatives: {fn} (missed drones)")
    
    if fp > fn:
        print(f"  → Model too sensitive (over-predicting drone)")
    elif fn > fp:
        print(f"  → Model too conservative (missing drones)")
    else:
        print(f"  → Balanced error distribution ✓")
    
    # Save model
    model.save_model(output_model)
    print(f"\n💾 Model saved to: {output_model}")
    
    # Save feature names
    with open('feature_names.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"💾 Feature names saved to: feature_names.json")
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE".center(80))
    print("="*80)
    
    print(f"\n🎯 Final Performance:")
    print(f"  Test Accuracy:  {accuracy:.1%}")
    print(f"  Test F1 Score:  {f1:.3f}")
    
    if accuracy >= 0.90:
        print(f"\n✅ EXCELLENT! Model performing very well!")
    elif accuracy >= 0.85:
        print(f"\n✅ GOOD! Model performing well.")
    elif accuracy >= 0.80:
        print(f"\n⚠️  ACCEPTABLE. Consider tuning hyperparameters.")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT. Try hyperparameter tuning or collect more data.")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Test on new audio files")
    print(f"  2. Monitor false positives/negatives")
    print(f"  3. Collect more data where model fails")
    print()
    
    return model, accuracy, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost (simplified, no plotting issues)')
    parser.add_argument('--features', required=True, help='Path to features.csv')
    parser.add_argument('--output', default='drone_detector_xgb.model', help='Output model file')
    
    args = parser.parse_args()
    
    model, accuracy, f1 = train_xgboost_simple(args.features, args.output)
    
    print(f"✅ Done! Model accuracy: {accuracy:.1%}, F1: {f1:.3f}")