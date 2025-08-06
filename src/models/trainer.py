import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from src.data.data_loader import load_data
from src.data.data_preprocessor import preprocess_data
from src.models.model import create_model, save_model
from src.utils.logger import setup_logger
import pandas as pd

logger = setup_logger()

def optimize_ridge(X_train, y_train):
    """Optimize Ridge Regression hyperparameters with Optuna."""
    def objective(trial):
        params = {'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True)}
        ridge = create_model('Ridge Regression', **params)
        score = cross_val_score(ridge, X_train, y_train, scoring='neg_mean_absolute_error', cv=5)
        return -score.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    return {'alpha': study.best_params['alpha']}

def train_models(data_path, test_path=None):
    """Train multiple models and log with MLflow."""
    mlflow.set_experiment("House_Price_Prediction")
    
    df_train = load_data(data_path)
    df_test = load_data(test_path) if test_path else None
    test_ids = df_test['Id'] if df_test is not None else None
    X, y, X_test, test_ids = preprocess_data(df_train, df_test, test_ids=test_ids)
    print(f"Number of features in X: {X.shape[1]}")
    print(f"Feature names: {X.columns.tolist()}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': {'model': create_model('Linear Regression')},
        'Lasso Regression': {'model': create_model('Lasso Regression', alpha=0.1)},
        'Ridge Regression': {'model': create_model('Ridge Regression', alpha=1.0)},
        'Random Forest Regression': {'model': create_model('Random Forest Regression', n_estimators=100, random_state=42)},
        'XGBoost Regression': {'model': create_model('XGBoost Regression', random_state=42)}
    }

    logger.info("Optimizing Ridge Regression hyperparameters")
    best_params = optimize_ridge(X_train, y_train)
    models['Ridge Regression']['model'] = create_model('Ridge Regression', **best_params)

    for name, config in models.items():
        with mlflow.start_run(run_name=name):
            model = config['model']
            logger.info(f"Training {name}")
            model.fit(X_train, y_train)
            save_model(model, f'{name.lower().replace(" ", "_")}_model.pkl')
            
            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)
            y_val_orig = np.expm1(y_val)
            
            mse = mean_squared_error(y_val_orig, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_orig, y_pred)
            r2 = r2_score(y_val_orig, y_pred)
            
            mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R2": r2})
            mlflow.sklearn.log_model(model, f"{name.lower().replace(' ', '_')}_model")
            logger.info(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

            if name == 'Ridge Regression' and X_test is not None and test_ids is not None:
                generate_submission(model, X_test, test_ids)

    return models

def generate_submission(model, X_test, test_ids):
    """Generate submission file."""
    y_test_pred_log = model.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)

    submission = pd.DataFrame({
        'Id': test_ids.iloc[:len(y_test_pred)],  
        'SalePrice': y_test_pred
    })
    submission.to_csv('submission.csv', index=False)

    logger.info("Submission file generated: submission.csv")
