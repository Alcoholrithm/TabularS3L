from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import Any, Union, Tuple
from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig
from .embedding import TS3LEmbeddingModule
from .backbone import TS3LBackboneModule

class TS3LModule(ABC, nn.Module):
    def __init__(self, embedding_config: BaseEmbeddingConfig, backbone_config: BaseBackboneConfig) -> None:
        super(TS3LModule, self).__init__()
        self.embedding_module = TS3LEmbeddingModule(embedding_config)
        self._set_backbone_module(backbone_config)
        
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        def init_decorator(cls_init):
            def new_init(self, *args, **kwargs):
                cls_init(self, *args, **kwargs)
                cls.set_first_phase(self)
            return new_init
        cls.__init__ = init_decorator(cls.__init__) # type: ignore
        
    def _set_backbone_module(self, backbone_config):
        self.backbone_module = TS3LBackboneModule(backbone_config)
        
    @property
    @abstractmethod
    def encoder(self):
        raise NotImplementedError
    
    def set_first_phase(self):
        """Set first phase step as the forward pass
        """
        self.forward = self._first_phase_step
        self.encoder.requires_grad_(True)
    
    def set_second_phase(self, freeze_encoder: bool = True):
        """Set second phase step as the forward pass
        """
        self.forward = self._second_phase_step
        self.encoder.requires_grad_(not freeze_encoder)
    
    @abstractmethod
    def _first_phase_step(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
    @abstractmethod
    def _second_phase_step(self, *args: Any, **kwargs: Any) -> Any:
        pass