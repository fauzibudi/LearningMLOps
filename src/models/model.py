from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

def create_model(model_type, **params):
    """Create a model based on type with filtered parameters."""
    model_params = {
        'Linear Regression': {},
        'Lasso Regression': {'alpha': params.get('alpha', 0.1)},
        'Ridge Regression': {'alpha': params.get('alpha', 1.0)},
        'Random Forest Regression': {'n_estimators': params.get('n_estimators', 100), 'random_state': params.get('random_state', 42)},
        'XGBoost Regression': {'random_state': params.get('random_state', 42)}
    }
    
    models = {
        'Linear Regression': LinearRegression(**model_params['Linear Regression']),
        'Lasso Regression': Lasso(**model_params['Lasso Regression']),
        'Ridge Regression': Ridge(**model_params['Ridge Regression']),
        'Random Forest Regression': RandomForestRegressor(**model_params['Random Forest Regression']),
        'XGBoost Regression': XGBRegressor(**model_params['XGBoost Regression'])
    }
    return models.get(model_type, Ridge(**model_params['Ridge Regression']))

def save_model(model, file_path):
    """Save the trained model."""
    joblib.dump(model, file_path)