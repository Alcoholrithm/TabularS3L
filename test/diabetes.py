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

    for col in category_cols:
        data[col] = le.fit_transform(data[col])

    temp = None
    for col in category_cols:
        oh_values = OneHotEncoder().fit_transform(data[col].values.reshape((-1, 1))).toarray()
        new_cols = [col + "-" + str(i) for i in range(len(data[col].unique()))]
        oh_values = pd.DataFrame(oh_values, columns = new_cols, dtype=np.int8, index=data.index)
        if temp is None:
            temp = oh_values
        else:
            temp = temp.merge(oh_values, left_index=True, right_index=True)

    data = data.merge(temp, left_index=True, right_index=True)
    data.drop(category_cols, inplace=True, axis=1)

    category_cols = temp.columns

    scaler = MinMaxScaler()
    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])
    
    return data, label, continuous_cols, category_cols