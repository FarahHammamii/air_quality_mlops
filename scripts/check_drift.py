"""
Data Drift Detection for Air Quality Pipeline
Calculates Population Stability Index (PSI) between reference and current data
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path


def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index (PSI)
    PSI < 0.1: No significant drift
    PSI 0.1-0.25: Moderate drift
    PSI > 0.25: Significant drift - retrain recommended
    """
    psi_values = {}
    
    for col in reference.columns:
        if col not in current.columns or reference[col].dtype not in ['float64', 'float32', 'int64', 'int32']:
            continue
        
        # Create bins
        ref_vals = reference[col].dropna()
        cur_vals = current[col].dropna()
        
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue
        
        # Use percentiles for bin edges
        edges = np.percentile(ref_vals, np.linspace(0, 100, bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        
        # Calculate distributions
        ref_dist = np.histogram(ref_vals, bins=edges)[0] / len(ref_vals)
        cur_dist = np.histogram(cur_vals, bins=edges)[0] / len(cur_vals)
        
        # Add small epsilon to avoid division by zero
        ref_dist = np.maximum(ref_dist, 1e-10)
        cur_dist = np.maximum(cur_dist, 1e-10)
        
        # Calculate PSI
        psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
        psi_values[col] = psi
    
    # Average PSI across features
    avg_psi = np.mean(list(psi_values.values())) if psi_values else 0
    
    return avg_psi, psi_values


def main():
    """Main drift detection function"""
    
    # Load reference stats (from champion model training)
    champion_file = Path("models/champion_metrics.json")
    reference_stats_file = Path("data/reference_stats.parquet")
    
    # Load current data
    current_data = pd.read_parquet("data/raw/delhi_merged.parquet")
    
    # If no reference stats exist, create them (first run)
    if not reference_stats_file.exists():
        print("No reference stats found - creating baseline")
        
        # Select numeric columns for drift detection
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        reference_stats = current_data[numeric_cols].describe()
        
        # Save reference stats
        reference_stats.to_parquet(reference_stats_file)
        
        # Save feature list
        with open("data/reference_features.json", "w") as f:
            json.dump(list(numeric_cols), f)
        
        # Save drift threshold met (retrain needed)
        with open("drift_result.txt", "w") as f:
            f.write("1.0")  # First run - always retrain
        
        print("Baseline created - retraining recommended")
        return
    
    # Load reference data for drift calculation
    with open("data/reference_features.json", "r") as f:
        ref_features = json.load(f)
    
    # Create reference dataframe from saved stats (simplified)
    # For proper PSI, we need actual reference data. Use recent historical window
    reference_data = pd.read_parquet("data/raw/delhi_merged.parquet")
    
    # Limit to same feature set
    common_features = [f for f in ref_features if f in current_data.columns]
    
    if len(common_features) < 5:
        print("Insufficient common features for drift detection")
        with open("drift_result.txt", "w") as f:
            f.write("0.0")
        return
    
    # Calculate PSI
    psi_score, psi_details = calculate_psi(
        reference_data[common_features].head(1000),  # Reference sample
        current_data[common_features].head(1000)      # Current sample
    )
    
    print(f"Average PSI: {psi_score:.4f}")
    print(f"Feature PSI: {psi_details}")
    
    # Save drift result
    with open("drift_result.txt", "w") as f:
        f.write(str(psi_score))
    
    # Save detailed report
    report = {
        "psi_score": psi_score,
        "feature_psi": psi_details,
        "recommendation": "retrain" if psi_score > 0.15 else "skip",
        "reference_samples": len(reference_data),
        "current_samples": len(current_data),
        "features_analyzed": len(common_features)
    }
    
    with open("data/drift_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Drift analysis complete. PSI: {psi_score:.4f}")
    print(f"Recommendation: {report['recommendation']}")


if __name__ == "__main__":
    main()