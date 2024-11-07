import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, btc_path, fgi_path):
        self.btc_path = btc_path
        self.fgi_path = fgi_path
        self.scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        
    def load_and_preprocess(self):
        # Load Bitcoin data
        btc_data = pd.read_csv(self.btc_path)
        btc_data['Date'] = pd.to_datetime(btc_data['Date'])
        btc_data.set_index('Date', inplace=True)
        
        # Load FGI data
        fgi_data = pd.read_csv(self.fgi_path)
        fgi_data['Date'] = pd.to_datetime(fgi_data['timestamp'], unit='s')
        fgi_data.set_index('Date', inplace=True)
        fgi_data = fgi_data[['value']].rename(columns={'value': 'FGI'})
        
        # Merge datasets
        data = pd.merge(btc_data, fgi_data, left_index=True, right_index=True, how='inner')
        
        # Engineer features
        data = self._engineer_features(data)
        
        # Drop any remaining NaN values
        data.dropna(inplace=True)
        
        return data
    
    def _engineer_features(self, data):
        # Price-based features
        data['Returns'] = data['Close'].pct_change()
        data['MA7'] = data['Close'].rolling(window=7).mean()
        data['MA30'] = data['Close'].rolling(window=30).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # Volume features
        data['VolMA7'] = data['Volume'].rolling(window=7).mean()
        data['Volume_Change'] = data['Volume'].pct_change()
        
        # Volatility
        data['Volatility'] = data['Returns'].rolling(window=30).std()
        
        # Price range features
        data['Daily_Range'] = (data['High'] - data['Low']) / data['Open']
        
        # FGI features
        data['FGI_MA7'] = data['FGI'].rolling(window=7).mean()
        data['FGI_Lag1'] = data['FGI'].shift(1)
        data['FGI_Lag7'] = data['FGI'].shift(7)
        
        return data
    
    def prepare_data(self, data, sequence_length=10):
        # Select features
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'Returns', 'MA7', 'MA30', 'MA50', 'MA200',
                   'VolMA7', 'Volume_Change', 'Volatility',
                   'Daily_Range', 'FGI', 'FGI_MA7', 'FGI_Lag1', 'FGI_Lag7']
        
        # Scale features
        scaled_data = self.scaler.fit_transform(data[features])
        scaled_df = pd.DataFrame(scaled_data, columns=features, index=data.index)
        
        # Separately scale Close prices for inverse transformation later
        self.price_scaler.fit_transform(data[['Close']])
        
        # Create sequences
        X, y = self._create_sequences(scaled_df, sequence_length)
        
        # Split data
        return train_test_split(X, y, test_size=0.2, shuffle=False)
    
    def _create_sequences(self, data, sequence_length):
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i:(i + sequence_length)].values)
            y.append(data.iloc[i + sequence_length]['Close'])
            
        return np.array(X), np.array(y)
    
    def prepare_future_sequence(self, data, sequence_length=10):
        # Select features
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'Returns', 'MA7', 'MA30', 'MA50', 'MA200',
                   'VolMA7', 'Volume_Change', 'Volatility',
                   'Daily_Range', 'FGI', 'FGI_MA7', 'FGI_Lag1', 'FGI_Lag7']
        
        # Get the last sequence_length rows
        latest_data = data[features].iloc[-sequence_length:].copy()
        
        # Scale the data using the fitted scaler
        scaled_data = self.scaler.transform(latest_data)
        
        return scaled_data