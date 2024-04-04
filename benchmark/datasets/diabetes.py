from sklearn.datasets import fetch_openml
import numpy as np
from types import SimpleNamespace
from typing import Tuple, List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def load_diabetes():
    
    diabetes = fetch_openml(data_id = 37, data_home='./data_cache')

    data = diabetes.data

    le = LabelEncoder()
    label = pd.Series(le.fit_transform(diabetes.target))

    category_cols = []
    continuous_cols = list(map(str, data.columns))

    scaler = MinMaxScaler()
    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])
    
    return data, label, continuous_cols, category_cols, 2, "accuracy_score", {}