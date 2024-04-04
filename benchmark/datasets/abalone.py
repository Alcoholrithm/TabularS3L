from sklearn.datasets import fetch_openml
import numpy as np
from types import SimpleNamespace
from typing import Tuple, List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def load_abalone():
    
    abalone = fetch_openml(data_id = 44956, data_home='./data_cache')

    data = abalone.data

    label = abalone.target

    
    category_cols = ["sex"]
    continuous_cols = []
    for col in data.columns:
        if not col in category_cols:
            continuous_cols.append(col)
    
    le = LabelEncoder()
    for col in category_cols:
        data[col] = le.fit_transform(data[col])
    
    scaler = MinMaxScaler()
    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])
    
    
    
    return data, label, continuous_cols, category_cols, 1, "mean_squared_error", {} 