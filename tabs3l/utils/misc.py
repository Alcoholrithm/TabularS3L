from sklearn import metrics
from abc import ABC, abstractmethod

class BaseScorer(ABC):
    def __init__(self, metric:str) -> None:
        """Base scorer for evaluation

        Args:
            metric (str): The name of the evaluation metric in sklearn.metrics
        """
        self.__name__ = metric
        self.metric = getattr(metrics, metric)
    
    @abstractmethod
    def __call__(self, y, y_hat) -> float:
        pass