from sklearn.datasets import fetch_openml
import numpy as np
from types import SimpleNamespace
from typing import Tuple, List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def load_cmc():
    cmc = fetch_openml(data_id = 23, data_home='./data_cache')

    data = cmc.data

    label = cmc.target
    
    label = label.astype(object)
    label.value = LabelEncoder().fit_transform(label)
    
    label = label.astype(int) - 1

    continuous_cols = ["Wifes_age", "Number_of_children_ever_born"]
    
    category_cols = []
    for col in data.columns:
        if not col in continuous_cols:
            category_cols.append(col)

    le = LabelEncoder()
    for col in category_cols:
        data[col] = le.fit_transform(data[col])
        
    return data, label, continuous_cols, category_cols, 3, "balanced_accuracy_score", {}