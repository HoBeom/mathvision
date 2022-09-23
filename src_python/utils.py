from enum import Enum
import numpy as np

class HomographyType(Enum):
    UNKNOWUN = -1
    NORMAL = 0
    CONCAVE = 1
    TWIST = 2
    REFLECTION = 3

    def __str__(self) -> str:
        return str(self.name)

def classifyHomography(pts1: np.ndarray, pts2: np.ndarray) -> int:
    if len(pts1) != 4 or len(pts2) != 4: 
        return HomographyType.UNKNOWUN

    return HomographyType.NORMAL

def polyArea(points):
    area = 0
    return area

