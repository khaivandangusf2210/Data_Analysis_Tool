import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import boxcox
import warnings
from typing import Tuple, Optional, List, Dict, Any
import streamlit as st
warnings.filterwarnings('ignore')


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    df_clean = df.copy()
    
    float32_cols = df_clean.select_dtypes(include=['float32']).columns
    if len(float32_cols) > 0:
        df_clean[float32_cols] = df_clean[float32_cols].astype('float64')
    
    for col in df_clean.columns:
        if any(date_indicator in col.upper() for date_indicator in ['DATE', 'TIME', 'YEAR', 'MONTH', 'DAY']):
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            except Exception:
                pass
    
    datetime_cols = df_clean.select_dtypes(include=['datetime64']).columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number, 'datetime64']).columns
    
    if strategy == 'mean':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif strategy == 'median':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif strategy == 'zero':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # handle categorical columns with mode
    for col in categorical_cols:
        mode_value = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "Unknown"
        df_clean[col] = df_clean[col].fillna(mode_value)
    
    # handle datetime columns 
    for col in datetime_cols:
        df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
    
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def encode_categorical(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    df_encoded = df.copy()
    
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    if target_col in categorical_cols:
        categorical_cols = categorical_cols.drop(target_col)
    
    if len(categorical_cols) == 0:
        return df_encoded
    
    # one-hot encoding for categorical columns
    try:
        ct = ColumnTransformer([
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ], remainder='passthrough')
        
        encoded_data = ct.fit_transform(df_encoded)
        
        onehot_features = ct.named_transformers_['onehot'].get_feature_names_out(categorical_cols)
        numeric_features = df_encoded.select_dtypes(include=[np.number]).columns
        feature_names = np.concatenate([onehot_features, numeric_features])
        
        df_transformed = pd.DataFrame(encoded_data, columns=feature_names, index=df_encoded.index)
        
        if target_col is not None and target_col in df.columns:
            df_transformed[target_col] = df_encoded[target_col]
        
        return df_transformed
    
    except Exception as e:
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        return df_encoded


def scale_features(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    df_scaled = df.copy()
    
    if target_col and target_col in df_scaled.columns:
        target = df_scaled[target_col]
        features = df_scaled.drop(columns=[target_col])
    else:
        target = None
        features = df_scaled
    
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        if target is not None:
            df_scaled[target_col] = target
        return df_scaled, None
    
    scaler = StandardScaler()
    
    scaled_features = features.copy()
    scaled_features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
    
    if target is not None:
        scaled_features[target_col] = target
    
    return scaled_features, scaler


def transform_skewed_features(df: pd.DataFrame, threshold: float = 0.5, target_col: Optional[str] = None) -> pd.DataFrame:
    df_transformed = df.copy()
    
    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
    if target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)
    
    for col in numeric_cols:
        skewness = df_transformed[col].skew()
        
        if abs(skewness) > threshold:
            if df_transformed[col].min() <= 0:
                shift = abs(df_transformed[col].min()) + 1
                df_transformed[col] = df_transformed[col] + shift
            
            try:
                df_transformed[col], _ = boxcox(df_transformed[col])
            except Exception:
                continue
    
    return df_transformed


def prepare_data_for_modeling(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df_copy = df.copy()
    
    if target_col in df_copy.columns:
        y = df_copy[target_col]
        X = df_copy.drop(columns=[target_col])
    else:
        y = None
        X = df_copy
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns
    
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    
    for col in categorical_cols:
        mode_value = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
        X[col] = X[col].fillna(mode_value)
    
    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes
    
    return X, y


def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    column_types = {
        'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'boolean': df.select_dtypes(include=['bool']).columns.tolist()
    }
    
    return column_types


def perform_analysis(data: pd.DataFrame, analysis_type: str):
    if analysis_type == "Statistical Summary":
        return {
            "title": "Statistical Summary",
            "description": "Basic statistical measures of your numerical variables",
            "result": data.describe()
        }
    elif analysis_type == "Missing Values Analysis":
        missing_data = pd.DataFrame({
            'Missing Values': data.isnull().sum(),
            'Percentage': (data.isnull().sum() / len(data) * 100).round(2)
        })
        return {
            "title": "Missing Values Analysis",
            "description": "Analysis of missing values in your dataset",
            "result": missing_data
        }
    elif analysis_type == "Correlation Analysis":
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            return {
                "title": "Correlation Analysis",
                "description": "Correlation matrix between numerical variables",
                "result": numeric_data.corr()
            }
        return {
            "title": "Correlation Analysis",
            "description": "Not enough numerical variables for correlation analysis",
            "result": None
        }
    elif analysis_type == "Distribution Analysis":
        numeric_data = data.select_dtypes(include=[np.number])
        distribution_stats = pd.DataFrame({
            'Mean': numeric_data.mean(),
            'Median': numeric_data.median(),
            'Std Dev': numeric_data.std(),
            'Skewness': numeric_data.skew(),
            'Kurtosis': numeric_data.kurtosis()
        })
        return {
            "title": "Distribution Analysis",
            "description": "Distribution statistics for numerical variables",
            "result": distribution_stats
        }
    elif analysis_type == "Outlier Detection":
        numeric_data = data.select_dtypes(include=[np.number])
        outlier_stats = pd.DataFrame(columns=['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound', 'Outliers'])
        
        for col in numeric_data.columns:
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = len(numeric_data[(numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)])
            
            outlier_stats.loc[col] = [Q1, Q3, IQR, lower_bound, upper_bound, outliers]
        
        return {
            "title": "Outlier Detection",
            "description": "Outlier analysis for numerical variables using IQR method",
            "result": outlier_stats
        }
    return None


@st.cache_data
def get_cached_analysis(data: pd.DataFrame, analysis_type: str):
    return perform_analysis(data, analysis_type)


def analyze_data_quality(data: pd.DataFrame):
    st.markdown("### Select Analyses to Perform")
    
    selected_analyses = st.multiselect(
        "Choose analyses to perform:",
        ["Statistical Summary", "Missing Values Analysis", "Correlation Analysis", 
         "Distribution Analysis", "Outlier Detection"],
        default=["Statistical Summary"]
    )
    
    if not selected_analyses:
        st.warning("Please select at least one analysis to perform.")
        return
    
    for analysis in selected_analyses:
        result = get_cached_analysis(data, analysis)
        if result:
            st.subheader(result["title"])
            st.write(result["description"])
            st.write(result["result"])


def preprocess_data(df):
    try:
        df = handle_missing_values(df)
        
        df = encode_categorical(df)
        
        df, _ = scale_features(df)
        
        df = transform_skewed_features(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return df 