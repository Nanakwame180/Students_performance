from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd

# 1. MOVE CONSTANTS TO TOP LEVEL
ORDINAL_COLS = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level']
NOMINAL_COLS = ['Extracurricular_Activities', 'Internet_Access']
ORDINAL_MAP = {'Low': 1, 'Medium': 2, 'High': 3}
NOMINAL_MAP = {'Yes': 1, 'No': 0}

# 2. MOVE FUNCTIONS TO TOP LEVEL
def ordinal_encode(df):
    df = df.copy()
    for col in ORDINAL_COLS:
        if col in df.columns:
            df[col] = df[col].map(ORDINAL_MAP)
    return df

def nominal_encode(df):
    df = df.copy()
    for col in NOMINAL_COLS:
        if col in df.columns:
            df[col] = df[col].map(NOMINAL_MAP)
    return df

def feature_engineering(df):
    df = df.copy()
    # Ensure columns exist before calculation to prevent errors
    if 'Hours_Studied' in df.columns and 'Sleep_Hours' in df.columns:
        df['Stress_Index'] = df['Hours_Studied'] / (df['Sleep_Hours'] + 1)
    
    if 'Motivation_Level' in df.columns and 'Access_to_Resources' in df.columns:
        df['Effective_Drive'] = df['Motivation_Level'] * df['Access_to_Resources']
        
    if 'Previous_Scores' in df.columns and 'Hours_Studied' in df.columns:
        df['Study_ROI'] = df['Previous_Scores'] / (df['Hours_Studied'] + 1)
    return df

# 3. KEEP ONLY THE PIPELINE BUILDING INSIDE THE FUNCTION
def get_preprocessor():
    pipeline = Pipeline([
        ("ordinal", FunctionTransformer(ordinal_encode)),
        ("nominal", FunctionTransformer(nominal_encode)),
        ("feature_eng", FunctionTransformer(feature_engineering)),
        ("scaler", StandardScaler())
    ])

    return pipeline