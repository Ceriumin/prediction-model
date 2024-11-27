from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np

class Model:
    def __init__(self, model_type='linear', params=None):
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            if params is None:
                params = {}
            self.model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        elif model_type == 'xgboost':
            self.model = XGBRegressor(objective='reg:squarederror', random_state=42)
        else:
            raise ValueError('Invalid model type')

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


    def evaluate(self, X_test, Y_test):
        return {
            'mse': mean_squared_error(Y_test, self.predict(X_test)),
            'r2': r2_score(Y_test, self.predict(X_test))
        }

    def tune_hyperparameters(self, X_train, Y_train, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, Y_train)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_












