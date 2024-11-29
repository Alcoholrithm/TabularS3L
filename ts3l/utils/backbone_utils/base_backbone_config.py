from dataclasses import dataclass, field


@dataclass
class BaseBackboneConfig:
    dropout_rate: float = field(default=0.3)