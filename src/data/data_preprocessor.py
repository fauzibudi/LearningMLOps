import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from src.utils.logger import setup_logger

logger = setup_logger()

def preprocess_data(df, df_test=None, target_column='SalePrice', test_ids=None):
    """Preprocess the dataset with error handling."""
    logger.info("Starting data preprocessing")
    
    # Hapus kolom Id dari data train dan test sebelum preprocessing
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    if df_test is not None and 'Id' in df_test.columns:
        df_test_original = df_test.copy()
        test_ids = df_test_original['Id']  # Simpan Id untuk submission
        df_test = df_test.drop('Id', axis=1)

    categorical_cols = [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
        'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
        'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
        'PoolQC', 'Fence', 'MiscFeature', 'MoSold', 'SaleType', 'SaleCondition'
    ]
    numerical_cols = [
        'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
        'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
        'MiscVal', 'YrSold'
    ]

    categorical_fill = {
        'PoolQC': 'No Pool', 'MiscFeature': 'None', 'Alley': 'No Alley', 
        'Fence': 'No Fence', 'MasVnrType': 'None', 'FireplaceQu': 'No Fireplace',
        'Exterior1st': 'Other', 'Exterior2nd': 'Other', 'GarageType': 'No Garage',
        'GarageFinish': 'No Garage', 'GarageQual': 'No Garage', 'GarageCond': 'No Garage',
        'BsmtExposure': 'No Basement', 'BsmtFinType2': 'No Basement', 
        'BsmtFinType1': 'No Basement', 'BsmtQual': 'No Basement', 'BsmtCond': 'No Basement'
    }
    numerical_fill_train = {
        'LotFrontage': df['LotFrontage'].median(),
        'GarageYrBlt': df['YearBuilt'],
        'MasVnrArea': 0.0,
        'GarageCars': 0,
        'GarageArea': 0,
        'BsmtHalfBath': 0, 'BsmtFullBath': 0,
        'BsmtFinSF1': 0, 'BsmtFinSF2': 0, 'BsmtUnfSF': 0, 'TotalBsmtSF': 0
    }
    numerical_fill_test = {
        'LotFrontage': df['LotFrontage'].median(),
        'GarageYrBlt': df['YearBuilt'],  # Gunakan YearBuilt dari train untuk konsistensi
        'MasVnrArea': 0.0,
        'GarageCars': 0,
        'GarageArea': 0,
        'BsmtHalfBath': 0, 'BsmtFullBath': 0,
        'BsmtFinSF1': 0, 'BsmtFinSF2': 0, 'BsmtUnfSF': 0, 'TotalBsmtSF': 0
    }

    df.fillna(categorical_fill, inplace=True)
    df.fillna(numerical_fill_train, inplace=True)
    df.dropna(inplace=True)

    if df_test is not None:
        df_test.fillna(categorical_fill, inplace=True)
        df_test.fillna(numerical_fill_test, inplace=True)
        original_indices = df_test.index
        df_test.dropna(subset=['MSZoning', 'SaleType', 'Functional', 'KitchenQual', 'Utilities', 'Electrical'], inplace=True)
        if test_ids is not None:
            test_ids = test_ids.loc[original_indices[df_test.index]]

    df = df[df['SalePrice'] < df['SalePrice'].quantile(0.99)]
    df = df[df['GrLivArea'] < df['GrLivArea'].quantile(0.99)]

    X = df.drop(target_column, axis=1)
    y = np.log1p(df[target_column])
    X_test_processed = df_test if df_test is not None else None

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X[col])
        X[col] = le.transform(X[col])
        if df_test is not None:
            X_test_processed[col] = X_test_processed[col].apply(lambda x: x if x in le.classes_ else 'Other')
            if 'Other' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Other')
            X_test_processed[col] = le.transform(X_test_processed[col])
        label_encoders[col] = le

    scaler = MinMaxScaler()
    scaler.fit(X[numerical_cols])
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    if df_test is not None:
        X_test_processed[numerical_cols] = scaler.transform(X_test_processed[numerical_cols])

    logger.info("Data preprocessing completed")
    return X, y, X_test_processed, test_ids