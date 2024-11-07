from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

class RandomForestModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train.ravel())
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        return self.model
    
    def predict(self, X):
        return self.model.predict(X) 