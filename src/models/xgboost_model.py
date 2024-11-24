import xgboost as xgb
from sklearn.model_selection import GridSearchCV

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            tree_method='hist'  # For faster training
        )
        
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error'
        )
        
        grid_search.fit(X_train, y_train.ravel())
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Print feature importance
        feature_importance = self.model.feature_importances_
        print("\nTop 10 Most Important Features:")
        for i in sorted(range(len(feature_importance)), 
                       key=lambda i: feature_importance[i], 
                       reverse=True)[:10]:
            print(f"Feature {i}: {feature_importance[i]:.4f}")
        
        return self.model
    
    def predict(self, X):
        return self.model.predict(X) 