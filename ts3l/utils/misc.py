from sklearn import metrics
import torchmetrics
    
class RegressionMetric(object):
    def __init__(self, metric, metric_hparams):
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
    def __init__(self, metric, metric_hparams):
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