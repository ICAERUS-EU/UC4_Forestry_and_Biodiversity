from enum import Enum, auto
from typing import List


class Band(Enum):
    AOT = 0  # R10m
    B01 = auto()  # R20m
    B02 = auto()  # R10m
    B03 = auto()  # R10m
    B04 = auto()  # R10m
    B05 = auto()  # R20m
    B06 = auto()  # R20m
    B07 = auto()  # R20m
    B08 = auto()  # R10m
    B09 = auto()  # R60m
    B11 = auto()  # R20m
    B12 = auto()  # R20m
    B8A = auto()  # R20m
    SCL = auto()  # R20m
    TCI = auto()  # R10m
    WVP = auto()  # R10m

    @classmethod
    def from_str(cls, string) -> "Band":
        return cls[string]

    def __str__(self):
        return self.to_str()

    def to_str(self) -> str:
        return self.name

    @classmethod
    def all_possible(cls) -> List[str]:
        return [band.name for band in cls]
