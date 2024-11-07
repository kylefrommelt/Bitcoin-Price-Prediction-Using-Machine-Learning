import torch
import numpy as np
from data_processor import DataProcessor
from models.lstm_model import LSTMPredictor, PyTorchTrainer
from models.random_forest_model import RandomForestModel
from evaluation import evaluate_models, plot_predictions

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize data processor
    processor = DataProcessor(
        btc_path='data/bitcoin_historical_yfinance.csv',
        fgi_path='data/fear_greed_index_data.csv'
    )
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = processor.load_and_preprocess()
    X_train, X_test, y_train, y_test = processor.prepare_data(data)
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    rf_model = RandomForestModel()
    rf_model.train(X_train.reshape(X_train.shape[0], -1), y_train)
    rf_predictions = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
    
    # Train LSTM model
    print("\nTraining LSTM model...")
    input_dim = X_train.shape[2]
    lstm_model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    )
    
    trainer = PyTorchTrainer(lstm_model)
    trainer.train(X_train, y_train, epochs=100)
    lstm_predictions = trainer.predict(X_test)
    
    # Evaluate models
    print("\nEvaluating models...")
    metrics = evaluate_models(y_test, rf_predictions, lstm_predictions, processor)
    
    # Plot results
    plot_predictions(y_test, rf_predictions, lstm_predictions, 
                    processor, dates=data.index[-len(y_test):])
    
    # Make future predictions
    print("\nMaking future predictions...")
    # Get the most recent sequence with the correct number of features
    latest_data = processor.prepare_future_sequence(data)
    
    # Make predictions
    rf_future = rf_model.predict(latest_data.reshape(1, -1))
    lstm_future = trainer.predict(latest_data.reshape(1, 10, -1))  # Reshape for LSTM
    
    # Inverse transform predictions
    rf_future = processor.price_scaler.inverse_transform(rf_future.reshape(-1, 1))[0][0]
    lstm_future = processor.price_scaler.inverse_transform(lstm_future.reshape(-1, 1))[0][0]
    
    print("\nFuture Price Predictions:")
    print(f"Random Forest: ${rf_future:,.2f}")
    print(f"LSTM: ${lstm_future:,.2f}")

if __name__ == "__main__":
    main() 