import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import os
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import warnings


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* General Body and Font */
    body {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        color: #E0E0E0; /* Light grey for text */
        background-color: #1A1A2E; /* Dark blue-purple background */
    }
    .stApp {
        background-color: #1A1A2E;
    }

    /* Main Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #6A0572 0%, #8E2DE2 50%, #4A00B0 100%); /* Deeper purple gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        padding-top: 1rem;
    }
    
    /* Section Title */
    .section-title {
        font-size: 2rem;
        font-weight: bold;
        color: #BB86FC; /* Light purple for titles */
        border-bottom: 3px solid #BB86FC;
        padding-bottom: 0.6rem;
        margin: 2rem 0 1.5rem 0;
        text-align: left;
    }
    
    /* Stat Cards */
    .stat-card {
        background: linear-gradient(135deg, #3A005F 0%, #6A0572 100%); /* Darker purple gradient */
        padding: 1.2rem;
        border-radius: 10px;
        color: #E0E0E0;
        text-align: center;
        margin: 0.6rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease-in-out;
    }
    .stat-card:hover {
        transform: translateY(-5px);
    }
    .stat-card h3 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        color: #BB86FC; /* Light purple for values */
    }
    .stat-card p {
        font-size: 1.1rem;
        color: #E0E0E0;
    }

    /* Info Panel */
    .info-panel {
        background-color: #2A003A; /* Darker purple for info panels */
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #BB86FC;
        margin: 1.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        color: #E0E0E0;
    }
    .info-panel h3 {
        color: #BB86FC;
        margin-top: 0;
    }
    .info-panel p {
        margin-bottom: 0.5rem;
    }

    /* Streamlit Widgets - Buttons, Selectboxes, Sliders */
    .stButton>button {
        background-color: #BB86FC; /* Light purple button */
        color: #1A1A2E; /* Dark text on button */
        border-radius: 8px;
        border: none;
        padding: 0.8rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #9C27B0; /* Darker purple on hover */
        color: #FFFFFF;
    }

    .stSelectbox>div>div, .stNumberInput>div>div, .stTextInput>div>div {
        background-color: #2A003A; /* Darker background for inputs */
        color: #E0E0E0;
        border-radius: 8px;
        border: 1px solid #BB86FC; /* Light purple border */
    }
    .stSelectbox div[data-baseweb="select"] > div, .stNumberInput div[data-baseweb="input"] > div, .stTextInput div[data-baseweb="input"] > div {
        background-color: #2A003A;
        color: #E0E0E0;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover, .stNumberInput div[data-baseweb="input"] > div:hover, .stTextInput div[data-baseweb="input"] > div:hover {
        border-color: #9C27B0; /* Darker purple on hover */
    }
    .stSelectbox div[data-baseweb="select"] > div:focus, .stNumberInput div[data-baseweb="input"] > div:focus, .stTextInput div[data-baseweb="input"] > div:focus {
        border-color: #9C27B0;
        box-shadow: 0 0 0 0.2rem rgba(187, 134, 252, 0.25);
    }

    /* Sidebar */
    .stSidebar {
        background-color: #2A003A; /* Darker purple for sidebar */
        color: #E0E0E0;
        padding-top: 2rem;
    }
    .stSidebar .stSelectbox label {
        color: #BB86FC; /* Light purple for sidebar selectbox label */
        font-weight: bold;
    }
    .stSidebar .stSelectbox div[data-baseweb="select"] > div {
        background-color: #3A005F; /* Even darker for sidebar selectbox */
        border: 1px solid #BB86FC;
    }
    .stSidebar .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #9C27B0;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #3A005F; /* Darker purple for expander header */
        color: #BB86FC;
        border-radius: 8px;
        padding: 0.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .streamlit-expanderContent {
        background-color: #2A003A;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #3A005F;
    }

    /* Text and Links */
    p, li {
        color: #E0E0E0;
        line-height: 1.6;
    }
    a {
        color: #BB86FC;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #BB86FC;
    }

    /* Plotly Chart Background */
    .js-plotly-plot {
        background-color: #1A1A2E !important;
    }
    .modebar {
        background-color: #1A1A2E !important;
        color: #E0E0E0 !important;
    }
    .modebar-btn {
        color: #E0E0E0 !important;
    }
    .modebar-btn:hover {
        background-color: #3A005F !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #3A005F;
    }

</style>
""", unsafe_allow_html=True)

COLOR_PALETTE = {
    'main': '#BB86FC',  
    'accent': '#CF6679', 
    'positive': '#03DAC6', 
    'caution': '#FEE440', 
    'neutral': '#03DAC6' 
}

API_BASE_URL = "http://localhost:8000"

def create_main_title(text):
    return f'<div class="main-title">{text}</div>'

def create_section_title(text):
    return f'<div class="section-title">{text}</div>'

def create_stat_card(value, label):
    return f'<div class="stat-card"><h3>{value}</h3><p>{label}</p></div>'

def create_info_panel(content):
    return f'<div class="info-panel">{content}</div>'

@st.cache_data
def load_data_csv(file_path):
    """Loads a CSV file from the specified path."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Please ensure the data files are in the 'data' directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading data from {file_path}: {e}")
        return None

FEATURE_DEFINITIONS = {
    "Id": "Unique identifier for each property.",
    "MSSubClass": "The type of dwelling involved in the sale.",
    "MSZoning": "Identifies the general zoning classification of the sale.",
    "LotFrontage": "Linear feet of street connected to property.",
    "LotArea": "Lot size in square feet.",
    "Street": "Type of road access to property.",
    "Alley": "Type of alley access to property.",
    "LotShape": "General shape of property.",
    "LandContour": "Flatness of the property.",
    "Utilities": "Type of utilities available.",
    "LotConfig": "Lot configuration.",
    "LandSlope": "Slope of property.",
    "Neighborhood": "Physical locations within Ames city limits.",
    "Condition1": "Proximity to main road or railroad.",
    "Condition2": "Proximity to main road or railroad (if a second is present).",
    "BldgType": "Type of dwelling.",
    "HouseStyle": "Style of dwelling.",
    "OverallQual": "Overall material and finish quality (1-10 scale).",
    "OverallCond": "Overall condition rating (1-10 scale).",
    "YearBuilt": "Original construction date.",
    "YearRemodAdd": "Remodel date (same as construction date if no remodel or addition).",
    "RoofStyle": "Type of roof.",
    "RoofMatl": "Roof material.",
    "Exterior1st": "Exterior covering on house.",
    "Exterior2nd": "Exterior covering on house (if more than one material).",
    "MasVnrType": "Masonry veneer type.",
    "MasVnrArea": "Masonry veneer area in square feet.",
    "ExterQual": "Exterior material quality.",
    "ExterCond": "Present condition of the material on the exterior.",
    "Foundation": "Type of foundation.",
    "BsmtQual": "Height of the basement.",
    "BsmtCond": "General condition of the basement.",
    "BsmtExposure": "Walkout or garden level basement walls.",
    "BsmtFinType1": "Quality of basement finished area.",
    "BsmtFinSF1": "Type 1 finished square feet.",
    "BsmtFinType2": "Quality of second finished area (if present).",
    "BsmtFinSF2": "Type 2 finished square feet.",
    "BsmtUnfSF": "Unfinished square feet of basement area.",
    "TotalBsmtSF": "Total square feet of basement area.",
    "Heating": "Type of heating.",
    "HeatingQC": "Heating quality and condition.",
    "CentralAir": "Central air conditioning.",
    "Electrical": "Electrical system.",
    "1stFlrSF": "First Floor square feet.",
    "2ndFlrSF": "Second floor square feet.",
    "LowQualFinSF": "Low quality finished square feet (all floors).",
    "GrLivArea": "Above grade (ground) living area square feet.",
    "BsmtFullBath": "Basement full bathrooms.",
    "BsmtHalfBath": "Basement half bathrooms.",
    "FullBath": "Full bathrooms above grade.",
    "HalfBath": "Half baths above grade.",
    "BedroomAbvGr": "Bedrooms above grade (does not include basement bedrooms).",
    "KitchenAbvGr": "Kitchens above grade.",
    "KitchenQual": "Kitchen quality.",
    "TotRmsAbvGrd": "Total rooms above grade (does not include bathrooms).",
    "Functional": "Home functionality (Sal = Severely Damaged --> Typ = Typical Functionality).",
    "Fireplaces": "Number of fireplaces.",
    "FireplaceQu": "Fireplace quality.",
    "GarageType": "Garage location.",
    "GarageYrBlt": "Year garage was built.",
    "GarageFinish": "Interior finish of the garage.",
    "GarageQual": "Garage quality.",
    "GarageCond": "Garage condition.",
    "PavedDrive": "Paved driveway.",
    "WoodDeckSF": "Wood deck area in square feet.",
    "OpenPorchSF": "Open porch area in square feet.",
    "EnclosedPorch": "Enclosed porch area in square feet.",
    "3SsnPorch": "Three season porch area in square feet.",
    "ScreenPorch": "Screen porch area in square feet.",
    "PoolArea": "Pool area in square feet.",
    "PoolQC": "Pool quality.",
    "Fence": "Fence quality.",
    "MiscFeature": "Miscellaneous feature not covered in other categories.",
    "MiscVal": "Value of miscellaneous feature.",
    "MoSold": "Month Sold (MM).",
    "YrSold": "Year Sold (YYYY).",
    "SaleType": "Type of sale.",
    "SaleCondition": "Condition of sale.",
    "SalePrice": "Sale price of the property (target variable)."
}

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

@st.cache_data
def preprocess_data(df_train_raw, df_test_raw):
    """
    Preprocesses the training and testing data, precisely replicating main.ipynb's logic.
    This includes replicating the specific (and potentially problematic) fillna behavior
    for numerical columns with Series values.
    """
    df_train = df_train_raw.copy()
    df_test = df_test_raw.copy()

    if 'Id' in df_train.columns:
        df_train = df_train.drop('Id', axis=1)
    if 'Id' in df_test.columns:
        df_test = df_test.drop('Id', axis=1)

    numerical_fill_train_series_dict = {
        'LotFrontage': df_train['LotFrontage'].median(),
        'GarageYrBlt': df_train['YearBuilt'], # This is a Series
        'MasVnrArea': 0.0,
        'GarageCars': 0,
        'GarageArea': 0,
        'BsmtHalfBath': 0, 'BsmtFullBath': 0,
        'BsmtFinSF1': 0, 'BsmtFinSF2': 0, 'BsmtUnfSF': 0, 'TotalBsmtSF': 0
    }

    numerical_fill_test_series_dict = {
        'LotFrontage': df_train['LotFrontage'].median(),  # Use training median
        'GarageYrBlt': df_test['YearBuilt'], # This is a Series
        'MasVnrArea': 0.0,
        'GarageCars': 0,
        'GarageArea': 0,
        'BsmtHalfBath': 0, 'BsmtFullBath': 0,
        'BsmtFinSF1': 0, 'BsmtFinSF2': 0, 'BsmtUnfSF': 0, 'TotalBsmtSF': 0
    }

    df_train.fillna(categorical_fill, inplace=True)
    df_train.fillna(numerical_fill_train_series_dict, inplace=True) 
    df_train.dropna(inplace=True) 

    df_test.fillna(categorical_fill, inplace=True)
    df_test.fillna(numerical_fill_test_series_dict, inplace=True) 
    df_test.dropna(subset=['MSZoning', 'SaleType', 'Functional', 'KitchenQual', 'Utilities', 'Electrical'], inplace=True)

    df_train = df_train[df_train['SalePrice'] < df_train['SalePrice'].quantile(0.99)]
    df_train = df_train[df_train['GrLivArea'] < df_train['GrLivArea'].quantile(0.99)]

    X = df_train.drop('SalePrice', axis=1)
    y = np.log1p(df_train['SalePrice']) 

    X_test = df_test.copy()

    train_cols = X.columns.tolist()
    test_cols = X_test.columns.tolist()

    for c in train_cols:
        if c not in test_cols:
            X_test[c] = 0
    
    for c in test_cols:
        if c not in train_cols:
            X_test = X_test.drop(c, axis=1)

    X_test = X_test[train_cols] 

    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            X[col] = le.transform(X[col])
            
            X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else 'Other')
            if 'Other' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Other')
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le

    scaler = MinMaxScaler()
    scaler.fit(X[numerical_cols])

    X[numerical_cols] = scaler.transform(X[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    numerical_fill_train_cached_scalars = {k: v.median() if isinstance(v, pd.Series) else v for k, v in numerical_fill_train_series_dict.items()}

    return X, y, X_test, label_encoders, scaler, categorical_cols, numerical_cols, numerical_fill_train_cached_scalars

@st.cache_resource
def train_and_tune_model(X, y):
    """
    Trains and tunes a Ridge Regression model using Optuna for hyperparameter optimization.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        """Optuna objective function for Ridge Regression."""
        alpha = trial.suggest_float('alpha', 0.1, 100.0, log=True)
        ridge = Ridge(alpha=alpha, random_state=42)
        score = cross_val_score(ridge, X_train, y_train, scoring='neg_mean_absolute_error', cv=5)
        return -score.mean()

    st.info("Starting Optuna hyperparameter tuning for Ridge Regression... This may take a few moments.")
    try:
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        st.success("Optuna tuning complete!")
        st.write(f"**Best Hyperparameters for Ridge Regression:** {study.best_params}")
    except Exception as e:
        st.error(f"Error during Optuna optimization: {e}")
        st.stop()

    best_ridge = Ridge(**study.best_params, random_state=42)
    best_ridge.fit(X_train, y_train)

    y_pred_val_log = best_ridge.predict(X_val)
    y_pred_val = np.expm1(y_pred_val_log) 
    y_val_orig = np.expm1(y_val) 

    st.subheader("Optimized Ridge Regression Performance on Validation Set:")
    mae = mean_absolute_error(y_val_orig, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_val))
    r2 = r2_score(y_val_orig, y_pred_val)

    col_metrics_1, col_metrics_2, col_metrics_3 = st.columns(3)
    with col_metrics_1:
        st.markdown(create_stat_card(f"{mae:,.2f}", "Mean Absolute Error (MAE)"), unsafe_allow_html=True)
    with col_metrics_2:
        st.markdown(create_stat_card(f"{rmse:,.2f}", "Root Mean Squared Error (RMSE)"), unsafe_allow_html=True)
    with col_metrics_3:
        st.markdown(create_stat_card(f"{r2:.4f}", "R-squared (R²)"), unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x=y_val_orig, y=y_pred_val, alpha=0.6, ax=ax, color=COLOR_PALETTE['main'])
    ax.plot([y_val_orig.min(), y_val_orig.max()], [y_val_orig.min(), y_val_orig.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Sale Price ($)', fontsize=12, color='#E0E0E0')
    ax.set_ylabel('Predicted Sale Price ($)', fontsize=12, color='#E0E0E0')
    ax.set_title('Optimized Ridge Regression: Actual vs Predicted (Validation Set)', fontsize=14, color='#BB86FC')
    ax.tick_params(axis='x', colors='#E0E0E0')
    ax.tick_params(axis='y', colors='#E0E0E0')
    ax.set_facecolor('#1A1A2E') # Plot background
    fig.patch.set_facecolor('#1A1A2E') # Figure background
    ax.spines['bottom'].set_color('#E0E0E0')
    ax.spines['top'].set_color('#E0E0E0')
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['right'].set_color('#E0E0E0')
    ax.legend()
    st.pyplot(fig)

    return best_ridge, X_train, y_train, X_val, y_val

st.sidebar.title("🌟 Navigation Menu")
page = st.sidebar.selectbox(
    "Select a Section:",
    ["👤 About Me", "📂 My Projects", "📈 Data Insights", "📉 Model Evaluation", "🔮 Prediction"]
)

if page == "👤 About Me":
    st.markdown(create_main_title("Fauzi Budi Wicaksono"), unsafe_allow_html=True)
    st.markdown(create_section_title("Data Science, AI ML Enthusiast"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(create_info_panel(f"""
        <h3>📞 Contact Details</h3>
        <p><strong>Email:</strong> fbwmalang123@gmail.com</p>
        <p><strong>Phone:</strong> <a href="wa.me/6285785068026" target="_blank" style="color: {COLOR_PALETTE['main']};">+62 85785068026 </a></p> 
        <p><strong>Location:</strong> Malang, East Java</p>
        <p><strong>GitHub:</strong> <a href="https://github.com/fauzibudi" target="_blank" style="color: {COLOR_PALETTE['main']};">github.com/fauzibudi</a></p>
        <p><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/fbudiw98/" target="_blank" style="color: {COLOR_PALETTE['main']};">linkedin.com/in/fbudiw98/</a></p>
        <p><strong>Portfolio Website:</strong> <a href="https://fauzibudi.github.io/web/" target="_blank" style="color: {COLOR_PALETTE['main']};">fauzibudi.github.io/web/</a></p>
        """), unsafe_allow_html=True)
    
    st.markdown(create_section_title("About Me"), unsafe_allow_html=True)
    st.write("""
    Career switcher with an academic background in Physics, equipped with skills in Python, SQL, Pandas, Scikit-learn, and Tableau. 
    Experienced in data analysis, visualization, and machine learning through hands-on projects completed during an AI & Machine Learning bootcamp. 
    Capable of processing and interpreting data to build predictive models and create visual dashboards that support data-driven decision-making. 
    Ready to contribute in data-related roles by applying strong analytical thinking, technical proficiency, and a problem-solving mindset to deliver valuable insights.
    """)
    
    st.markdown(create_section_title("Skills"), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"""
        **Programming & ML**
        - Python, SQL
        - Scikit-learn, Tensorflow
        - Pandas, NumPy
        """)
    with col2:
        st.write(f"""
        **Visualization & Tools**
        - Matplotlib, Plotly, Seaborn
        - Docker, Tableau
        - Streamlit
        """)
    st.markdown(create_section_title("Education"), unsafe_allow_html=True)
    st.write("""
    **Bachelor Degree of Physics** | University of Brawijaya  
    """)
    st.write("""
    **AI ML Bootcamp** | Dibimbing  
    """)
    st.write("""
    **Data Science Training** | Digital Talent Scholarship 
    """)

elif page == "📂 My Projects":
    st.markdown(create_main_title("My Other Projects"), unsafe_allow_html=True)
    st.markdown(create_section_title("Project Highlights"), unsafe_allow_html=True)
    st.write("""
    Feel free to explore my portfolio! Here, you'll find projects that showcase my growing skills and dedication to learning. As I continue to develop, I look forward to creating impactful solutions and sharing my journey with the data science community.
    """)
    
    projects = [
        {
            "name": "Smart Butterfly Classification: Leveraging AI to Recognize Nature's Beauty",
            "desc": "This project develops an intelligent system based on CNN and Transfer Learning for accurate butterfly classification, with the model optimized through tuning to achieve a validation accuracy of 0.86.",
            "link": "https://butterflypred.streamlit.app/" # Replace with actual link
        },
        {
            "name": "RAG Indonesian Independence History Project",
            "desc": "An AI-based application that is able to answer questions about Indonesian Independence history by presenting Wikipedia reference sources, combining RAG (Retrieval-Augmented Generation) technology.",
            "link": "https://github.com/fauzibudi/projects/tree/main/day-124-RAGSystem" # Replace with actual link
        },
        {
            "name": "AI Ethical Text Generation Application (GPT-2 Fine-tuned)",
            "desc": "This project involves fine-tuning the GPT-2 model on an AI ethics dataset to generate relevant text, optimized with Optuna.",
            "link": "https://huggingface.co/spaces/Hokeno/EthicalAITextGenerator" # Replace with actual link
        }
    ]
    for proj in projects:
        st.subheader(proj["name"])
        st.write(proj["desc"])
        st.markdown(f"[View My Project]({proj['link']})")
        st.markdown("---")

elif page == "📈 Data Insights":
    st.markdown(create_main_title("Housing Data Exploration"), unsafe_allow_html=True)
    df_raw = load_data_csv('data/train.csv')
    
    if df_raw is None:
        st.warning("Data not loaded. Please ensure 'data/train.csv' exists.")
    else:
        st.markdown(create_section_title("Dataset Overview"), unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(create_stat_card(len(df_raw), "Total Houses"), unsafe_allow_html=True)
        with col2:
            avg_price = df_raw['SalePrice'].mean()
            st.markdown(create_stat_card(f"${avg_price:,.0f}", "Avg Price"), unsafe_allow_html=True)
        with col3:
            max_price = df_raw['SalePrice'].max()
            st.markdown(create_stat_card(f"${max_price:,.0f}", "Max Price"), unsafe_allow_html=True)
        
        st.markdown(create_section_title("Raw Data Sample"), unsafe_allow_html=True)
        st.dataframe(df_raw.head(), use_container_width=True)
        st.write(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

        st.markdown(create_section_title("Feature Definitions"), unsafe_allow_html=True)
        selected_feature = st.selectbox("Select a feature to see its definition:", list(FEATURE_DEFINITIONS.keys()))
        if selected_feature:
            st.info(FEATURE_DEFINITIONS[selected_feature])

        st.markdown(create_section_title("Sale Price Distribution"), unsafe_allow_html=True)
        fig_price_dist = make_subplots(rows=1, cols=2, subplot_titles=("Original Sale Price Distribution", "Log-Transformed Sale Price Distribution"))

        fig_price_dist.add_trace(go.Histogram(x=df_raw['SalePrice'], name='SalePrice', marker_color=COLOR_PALETTE['main']), row=1, col=1)
        fig_price_dist.add_trace(go.Histogram(x=np.log1p(df_raw['SalePrice']), name='Log(SalePrice)', marker_color=COLOR_PALETTE['accent']), row=1, col=2)

        fig_price_dist.update_layout(
            height=400, showlegend=False,
            plot_bgcolor='#1A1A2E', paper_bgcolor='#1A1A2E',
            font=dict(color='#E0E0E0'),
            xaxis=dict(title_text="Sale Price", showgrid=False),
            xaxis2=dict(title_text="Log(Sale Price)", showgrid=False),
            yaxis=dict(title_text="Count", showgrid=False),
            yaxis2=dict(title_text="Count", showgrid=False)
        )
        st.plotly_chart(fig_price_dist, use_container_width=True)
        st.info("The SalePrice distribution is right-skewed, which is common for prices. Log-transformation helps normalize it for better model performance.")

        st.markdown(create_section_title("Missing Values Analysis"), unsafe_allow_html=True)
        missing_train = df_raw.isnull().sum()
        missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
        
        if not missing_train.empty:
            fig_missing = px.bar(
                missing_train,
                x=missing_train.index,
                y=missing_train.values,
                title="Missing Values Count in Training Data",
                labels={'x': 'Feature', 'y': 'Missing Count'},
                color_discrete_sequence=[COLOR_PALETTE['accent']]
            )
            fig_missing.update_layout(
                plot_bgcolor='#1A1A2E', paper_bgcolor='#1A1A2E',
                font=dict(color='#E0E0E0'),
                xaxis=dict(tickangle=-45, showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_missing, use_container_width=True)
            st.info("Missing values are handled during preprocessing by imputation.")
        else:
            st.success("No missing values found in the raw training data (after initial load).")

        st.markdown(create_section_title("Top Correlated Features with SalePrice"), unsafe_allow_html=True)
        numerical_df_for_corr = df_raw[numerical_cols + ['SalePrice']].copy()
        corr_matrix = numerical_df_for_corr.corr()
        
        if 'SalePrice' in corr_matrix.columns:
            top_corr = corr_matrix['SalePrice'].sort_values(ascending=False).drop('SalePrice').head(10)
            fig_top_corr = px.bar(
                top_corr,
                x=top_corr.index,
                y=top_corr.values,
                title="Top 10 Features Correlated with SalePrice",
                labels={'x': 'Feature', 'y': 'Correlation Coefficient'},
                color_discrete_sequence=[COLOR_PALETTE['positive']]
            )
            fig_top_corr.update_layout(
                plot_bgcolor='#1A1A2E', paper_bgcolor='#1A1A2E',
                font=dict(color='#E0E0E0'),
                xaxis=dict(tickangle=-45, showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_top_corr, use_container_width=True)
            st.info("Features like GarageArea, GrLivArea, and GarageCars show strong positive correlation with SalePrice.")
        else:
            st.warning("SalePrice column not found for correlation analysis.")

elif page == "📉 Model Evaluation":
    st.markdown(create_main_title("Model Performance"), unsafe_allow_html=True)
    st.markdown(create_section_title("Model Training & Evaluation"), unsafe_allow_html=True)

    df_train_raw, df_test_raw = load_data_csv('data/train.csv'), load_data_csv('data/test.csv')
    if df_train_raw is None or df_test_raw is None:
        st.warning("Cannot proceed with model evaluation. Please ensure data files are available.")
    else:
        original_test_ids = df_test_raw['Id']

        with st.spinner("Preprocessing data for model training..."):
            X, y, X_test_processed, label_encoders, scaler, _, _, numerical_fill_train_cached = preprocess_data(df_train_raw, df_test_raw)
        st.success("Data preprocessed!")

        if st.button("Train and Tune Ridge Regression Model"):
            with st.spinner("Training and tuning the model... This might take a moment."):
                best_ridge_model, X_train, y_train, X_val, y_val = train_and_tune_model(X, y)
            st.session_state['best_ridge_model'] = best_ridge_model
            st.session_state['X_test_processed'] = X_test_processed
            st.session_state['test_ids'] = original_test_ids # Store original test IDs
            st.session_state['label_encoders'] = label_encoders
            st.session_state['scaler'] = scaler
            st.session_state['numerical_fill_train_cached'] = numerical_fill_train_cached
            st.session_state['X_columns'] = X.columns.tolist() # Store column order
            st.success("Model training and tuning complete and results displayed above!")
        else:
            st.info("Click the button above to train and evaluate the Ridge Regression model.")

        if 'best_ridge_model' in st.session_state:
            st.markdown(create_section_title("Model Persistence"), unsafe_allow_html=True)
            st.write("The trained model and preprocessing objects are cached for faster predictions.")
            st.info("You can now navigate to the 'Prediction' page to use the trained model.")
        else:
            st.info("Model not yet trained. Please click 'Train and Tune Ridge Regression Model' above.")

elif page == "🔮 Prediction":
    st.markdown(create_main_title("House Price Prediction"), unsafe_allow_html=True)
    st.markdown(create_section_title("Make Predictions"), unsafe_allow_html=True)

    if 'best_ridge_model' not in st.session_state:
        st.warning("Model not trained yet. Please go to 'Model Evaluation' page and train the model first.")
        st.stop() 

    best_ridge_model = st.session_state['best_ridge_model']
    X_test_processed = st.session_state['X_test_processed']
    test_ids = st.session_state['test_ids']
    label_encoders = st.session_state['label_encoders']
    scaler = st.session_state['scaler']
    numerical_fill_train_cached = st.session_state['numerical_fill_train_cached']
    X_columns = st.session_state['X_columns'] 

    st.write("The Ridge Regression model is trained and ready for predictions.")
    
    st.markdown(create_section_title("Batch Prediction on Test Data"), unsafe_allow_html=True)
    if st.button("Generate Submission File for Test Data"):
        with st.spinner("Generating predictions for the test set..."):
            y_test_pred_log = best_ridge_model.predict(X_test_processed)
            y_test_pred = np.expm1(y_test_pred_log) 

            submission = pd.DataFrame({
                'Id': test_ids.loc[X_test_processed.index],
                'SalePrice': y_test_pred
            })
            submission_path = 'submission.csv'
            submission.to_csv(submission_path, index=False)
        st.success(f"Submission file generated: {submission_path}")
        st.download_button(
            label="Download submission.csv",
            data=submission.to_csv(index=False).encode('utf-8'),
            file_name='submission.csv',
            mime='text/csv',
        )
        st.write("First 5 rows of the generated submission file:")
        st.dataframe(submission.head())

    st.markdown("---")
    st.markdown(create_section_title("Predict for a Single House (Interactive Input)"), unsafe_allow_html=True)
    st.write("Enter values for a hypothetical house to get a price prediction:")

    user_input = {}
    
    tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])

    with tab1:
        st.markdown("#### Numerical Features")
        for col in numerical_cols:
            default_value = numerical_fill_train_cached.get(col, 0)
            default_value = float(default_value) 

            if col in ['YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt']:
                user_input[col] = st.number_input(f"{FEATURE_DEFINITIONS.get(col, col)} ({col})", min_value=1800, max_value=2023, value=int(default_value) if default_value else 2000, key=f"num_{col}")
            elif col in ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']:
                user_input[col] = st.number_input(f"{FEATURE_DEFINITIONS.get(col, col)} ({col})", min_value=0, value=int(default_value) if default_value else 1000, key=f"num_{col}")
            elif col in ['LotFrontage']:
                user_input[col] = st.number_input(f"{FEATURE_DEFINITIONS.get(col, col)} ({col})", min_value=0.0, value=float(default_value) if default_value else 60.0, key=f"num_{col}")
            else:
                user_input[col] = st.number_input(f"{FEATURE_DEFINITIONS.get(col, col)} ({col})", min_value=0, value=int(default_value) if default_value else 0, key=f"num_{col}")

    with tab2:
        st.markdown("#### Categorical Features")
        for col in categorical_cols:
            if col in label_encoders:
                options = list(label_encoders[col].classes_)
                if 'Other' not in options and 'Other' in label_encoders[col].classes_:
                    options.append('Other')
                
                default_cat_val = categorical_fill.get(col, options[0] if options else '')
                if default_cat_val not in options and options:
                    default_cat_val = options[0] 
                
                user_input[col] = st.selectbox(f"{FEATURE_DEFINITIONS.get(col, col)} ({col})", options, index=options.index(default_cat_val) if default_cat_val in options else 0, key=f"cat_{col}")
            else:
                user_input[col] = st.text_input(f"{FEATURE_DEFINITIONS.get(col, col)} ({col})", "Unknown", key=f"cat_{col}")

    if st.button("Predict Single House Price", key="predict_single_button"):
        try:
            single_house_df = pd.DataFrame([user_input])
            single_house_df.fillna(categorical_fill, inplace=True)
            if 'GarageYrBlt' in single_house_df.columns:
                single_house_df['GarageYrBlt'] = single_house_df['GarageYrBlt'].fillna(single_house_df['YearBuilt'])
            
            for col, val in numerical_fill_train_cached.items():
                if col in single_house_df.columns:
                    single_house_df[col] = single_house_df[col].fillna(val)


            for col in categorical_cols:
                if col in label_encoders:
                    le = label_encoders[col]
                    single_house_df[col] = single_house_df[col].apply(lambda x: x if x in le.classes_ else 'Other')
                    single_house_df[col] = le.transform(single_house_df[col])
                else:
                    st.warning(f"LabelEncoder not found for {col}. Skipping encoding for this column.")
                    single_house_df[col] = pd.to_numeric(single_house_df[col], errors='coerce').fillna(0) 

            single_house_df[numerical_cols] = scaler.transform(single_house_df[numerical_cols])

            single_house_df = single_house_df[X_columns]

            predicted_log_price = best_ridge_model.predict(single_house_df)
            predicted_price = np.expm1(predicted_log_price)[0]

            st.success(f"Predicted House Price: **\${predicted_price:,.2f}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check your input values and try again. Detailed error:")
            st.exception(e) # Show full traceback for debugging
            st.write("Debug Info (Single House DF after preprocessing attempts):")
            st.dataframe(single_house_df)
            st.write("Debug Info (Expected X.columns):")
            st.write(X_columns)

st.markdown("---")
st.markdown(f"""
<div class="footer">
    <p>Developed by Fauzi Budi | House Price Prediction Tool</p>
</div>
""", unsafe_allow_html=True)


