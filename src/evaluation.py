import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(y_true, rf_pred, lstm_pred, processor=None):
    if processor is not None:
        y_true = processor.price_scaler.inverse_transform(y_true.reshape(-1, 1))
        rf_pred = processor.price_scaler.inverse_transform(rf_pred.reshape(-1, 1))
        lstm_pred = processor.price_scaler.inverse_transform(lstm_pred.reshape(-1, 1))
    
    models = {
        'Random Forest': rf_pred.flatten(),
        'LSTM': lstm_pred.flatten()
    }
    
    metrics = {}
    for name, predictions in models.items():
        metrics[name] = {
            'MAE': mean_absolute_error(y_true, predictions),
            'RMSE': np.sqrt(mean_squared_error(y_true, predictions)),
            'R2': r2_score(y_true, predictions)
        }
        
        print(f"\n{name} Performance:")
        for metric, value in metrics[name].items():
            print(f"{metric}: {value:.2f}")
            
    return metrics

def plot_predictions(y_true, rf_pred, lstm_pred, processor=None, dates=None):
    if processor is not None:
        y_true = processor.price_scaler.inverse_transform(y_true.reshape(-1, 1))
        rf_pred = processor.price_scaler.inverse_transform(rf_pred.reshape(-1, 1))
        lstm_pred = processor.price_scaler.inverse_transform(lstm_pred.reshape(-1, 1))
    
    plt.figure(figsize=(15, 7))
    
    if dates is None:
        dates = range(len(y_true))
    
    plt.plot(dates, y_true, label='Actual', color='black', alpha=0.7)
    plt.plot(dates, rf_pred, label='Random Forest', color='blue', alpha=0.5)
    plt.plot(dates, lstm_pred, label='LSTM', color='red', alpha=0.5)
    
    plt.title('Bitcoin Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show() 