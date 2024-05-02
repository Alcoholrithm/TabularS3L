from typing import Dict, Any, List
import pandas as pd
from sklearn import metrics
import torchmetrics
    
class RegressionMetric(object):
    def __init__(self, metric: str, metric_hparams: Dict[str, Any] = {}):
        """Regression Metric Wrapper for TabularS3L.

        Args:
            metric (str): A name of regression metric of sklearn.metrics or torchmetrics.functional.
            metric_hparams (_type_): A hyperparameters for the given metric. Default is an empty dictionary.
        """
        super(RegressionMetric, self).__init__()
        
        self.metric_hparams = metric_hparams
        
        if hasattr(metrics, metric):
            self.metric = getattr(metrics, metric)
            self.forward = self.sklearn_forward
        else:
            self.metric = getattr(torchmetrics.functional, metric)
            self.forward = self.torchmetrics_forward
            
        self.__name__ = self.metric.__name__
    
    def __call__(self, preds, target):
        return self.forward(preds, target)
    
    def sklearn_forward(self, preds, target):
        return self.metric(target, preds, **self.metric_hparams)

    def torchmetrics_forward(self, preds, target):
        return self.metric(preds, target, **self.metric_hparams)

class ClassificationMetric(object):
    def __init__(self, metric: str, metric_hparams: Dict[str, Any] = {}):
        """Classification Metric Wrapper for TabularS3L.

        Args:
            metric (str): A name of classification metric of sklearn.metrics or torchmetrics.functional.
            metric_hparams (_type_): A hyperparameters for the given metric. Default is an empty dictionary.
        """
        super(ClassificationMetric, self).__init__()
        
        self.metric_hparams = metric_hparams
        
        if hasattr(metrics, metric):
            self.metric = getattr(metrics, metric)
            self.forward = self.sklearn_forward
        else:
            self.metric = getattr(torchmetrics.functional, metric)
            self.forward = self.torchmetrics_forward
            
        self.__name__ = self.metric.__name__
    
    def __call__(self, preds, target):
        return self.forward(preds, target)
    
    def sklearn_forward(self, preds, target):
        return self.metric(target, preds.argmax(1), **self.metric_hparams)
    
    def torchmetrics_forward(self, preds, target):
        return self.metric(preds.argmax(1), target, **self.metric_hparams)
    
def get_category_dims(data: pd.DataFrame, category_cols: List[str]):
    """Calculate the number of unique values (dimensionality) for each categorical column specified in a dataset.

    This function iterates over each column specified as categorical in 'category_cols', determines
    the unique elements in each of those columns, and appends the count of these unique elements
    to a list, which it returns.
    
    Args:
        data (pd.DataFrame): The dataset containing the categorical columns.
        category_cols (List[str]): A list of column names in 'data' that are considered categorical.

    Returns:
        List[int]: A list containing the counts of unique values for each categorical column specified.
                    Each element in the list corresponds to the number of unique values in the respective
                    column in 'category_cols'.
    """
    category_dims = []
    for col in category_cols:
        category_dims.append(len(set(data[col].values)))
    return category_dims