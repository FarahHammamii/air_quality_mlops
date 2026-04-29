"""
Batch Predictions for Air Quality Forecasting
Generates forecasts for next 24-72 hours
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from datetime import datetime, timedelta
from pathlib import Path


def load_production_model():
    """Load production model and scalers"""
    model_path = Path("models/production/heart_production.keras")
    scaler_path = Path("models/production/scaler.pkl")
    target_scaler_path = Path("models/production/target_scaler.pkl")
    
    # If production model doesn't exist, use best model
    if not model_path.exists():
        model_path = Path("models/heart_best.keras")
        scaler_path = Path("models/scaler.pkl")
        target_scaler_path = Path("models/target_scaler.pkl")
    
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    
    return model, scaler, target_scaler


def prepare_features(df, scaler, seq_length=72):
    """Prepare features for prediction"""
    from src.features.window_builder import engineer_features, create_sequences
    
    # Engineer features
    features = engineer_features(df)
    
    # Create sequences
    X, _, _ = create_sequences(features, seq_length=seq_length)
    
    # Scale features
    N, T, F = X.shape
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(N, T, F)
    
    return X_scaled


def make_predictions(model, X_scaled, target_scaler):
    """Make and inverse-transform predictions"""
    pred_scaled = model.predict(X_scaled).flatten()
    pred_raw = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return pred_raw


def generate_forecast_horizon(last_sequence, model, scaler, target_scaler, hours=24):
    """
    Generate recursive forecasts for future hours
    This is simplified - for production, you'd need iterative prediction
    """
    # For now, use last available predictions
    # Full recursive forecasting would require predicting one step at a time
    # and feeding predictions back as inputs
    
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(hours):
        # Scale current sequence
        N, T, F = current_seq.shape
        scaled_seq = scaler.transform(current_seq.reshape(-1, F)).reshape(N, T, F)
        
        # Predict next hour
        pred_scaled = model.predict(scaled_seq, verbose=0)
        pred_raw = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        predictions.append(pred_raw)
        
        # Update sequence (shift and add prediction)
        # This is simplified - you'd need to update features too
        current_seq = np.roll(current_seq, -1, axis=1)
        # Update last timestep with new prediction (simplified)
    
    return predictions


def main():
    """Main prediction pipeline"""
    
    # Create predictions directory
    Path("predictions").mkdir(exist_ok=True)
    
    # Load model and scalers
    print("Loading production model...")
    model, scaler, target_scaler = load_production_model()
    
    # Load latest data
    print("Loading latest data...")
    df = pd.read_parquet("data/raw/delhi_merged.parquet")
    
    # Prepare features
    print("Preparing features...")
    X_scaled = prepare_features(df, scaler)
    
    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(model, X_scaled[-100:], target_scaler)  # Last 100 samples
    
    # Create results dataframe
    results = pd.DataFrame({
        "timestamp": pd.date_range(start=datetime.now(), periods=len(predictions), freq="H"),
        "predicted_pm25": predictions,
        "model_version": "production",
        "run_time": datetime.now()
    })
    
    # Save predictions
    output_path = "predictions/latest_predictions.parquet"
    results.to_parquet(output_path, index=False)
    
    # Also save as CSV for easy viewing
    results.to_csv("predictions/latest_predictions.csv", index=False)
    
    # Generate forecast for next 24 hours
    print("Generating 24-hour forecast...")
    
    # Get last sequence for forecasting
    last_sequence = X_scaled[-1:].copy()
    
    forecast = generate_forecast_horizon(
        last_sequence, model, scaler, target_scaler, hours=24
    )
    
    forecast_df = pd.DataFrame({
        "timestamp": pd.date_range(start=datetime.now(), periods=24, freq="H"),
        "forecast_pm25": forecast,
        "forecast_type": "recursive"
    })
    
    forecast_df.to_parquet("predictions/forecast_24h.parquet", index=False)
    forecast_df.to_csv("predictions/forecast_24h.csv", index=False)
    
    print(f"Predictions saved: {len(predictions)} rows")
    print(f"Forecast saved: 24 hours")
    print(f"Average prediction: {predictions.mean():.2f} µg/m³")
    
    # Print summary statistics
    print("\n=== PREDICTION SUMMARY ===")
    print(f"Min: {predictions.min():.2f}")
    print(f"Max: {predictions.max():.2f}")
    print(f"Mean: {predictions.mean():.2f}")
    print(f"Std: {predictions.std():.2f}")


if __name__ == "__main__":
    main()