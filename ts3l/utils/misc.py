from typing import Dict, Any
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