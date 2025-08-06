# House Price Prediction MLOps Project

This project implements an end-to-end MLOps pipeline for predicting house prices(Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview). It includes modular code, MLflow tracking, logging, FastAPI deployment, and Docker containerization. We using ridge model with MAE: 13880.88, RMSE: 19946.24, R-squared: 0.9102.

Setup Video: https://drive.google.com/file/d/11GEU9LT9-v98e_lSz-bzxNruTbWDvPW_/view?usp=sharing

## Setup
1. Set up venv `py -3.10 -m venv tf_env` 
`source tf_env/Scripts/activate`
2. Install dependencies `pip install -r requirements.txt`
3. Train model `python -c "from src.models.trainer import train_models; train_models('data/train.csv', 'data/test.csv')"`
4. Run Mlflow UI `mlflow ui`, view Mlflow : `http://localhost:5000`
5. Start Api Server `python main_api.py` and open `http://localhost:8000/`
6. Try Test Prediction (Optional) `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [0, 2, 0.202055, 0.048246, 1, 1, 3, 3, 0, 4, 0, 12, 1, 2, 0, 2, 4, 5, 0.644928, 0.183333, 1, 0, 12, 13, 2, 0, 3, 4, 1, 4, 4, 3, 5, 0.248936, 3, 0.097693, 0.115582, 0.275109, 1, 4, 1, 4, 0.256621, 0, 0, 0.221434, 0, 0, 0.333333, 0, 0.333333, 0.333333, 3, 0.3, 6, 0, 3, 1, 0.644928, 3, 0.25, 0.52518, 5, 5, 2, 0.163361, 0, 0, 0, 0.25, 0, 2, 2, 1, 0, 5, 1, 8, 4]}'`. This features is a test dataset row 1 and the result is around 121089
7. run docker `docker-compose up --build`
8. Run `http://localhost:5000` for MLflow in docker and `http://localhost:8000/` for FastAPI in docker

## Project Structure
```
│
├── data/                  # Contains train.csv and test.csv
├── src/
│   ├── data/              # Data loading and preprocessing scripts
│   ├── models/            # Model training and prediction scripts
│   ├── api.py             # FastAPI application
│   ├── utils/             # Utility functions (logger, config)
├── notebook/              # Contain notebook(.ipynb) for EDA and experiments
├── main_api.py            # Entry point for the API server
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── house_price_prediction.log  # Log file
└── README.md              # This file
```





