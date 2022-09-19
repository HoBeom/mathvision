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

    vector_pmp = pts1 - np.roll(pts1, -1, axis=0)
    vector_ppp = pts1 - np.roll(pts1, 1, axis=0)
    cross_p = np.cross(vector_pmp, vector_ppp)

    vector_qmq = pts2 - np.roll(pts2, -1, axis=0)
    vector_qqq = pts2 - np.roll(pts2, 1, axis=0)
    cross_q = np.cross(vector_qmq, vector_qqq)

    dot_p_q = cross_p * cross_q
    condition_dot_p_q = (dot_p_q < 0).sum()
    if condition_dot_p_q == 4:
        return HomographyType.REFLECTION
    elif condition_dot_p_q == 2:
        return HomographyType.TWIST
    elif condition_dot_p_q in [1, 3]:
        return HomographyType.CONCAVE
    return HomographyType.NORMAL

def polyArea(points):
    # 일반 배열로 입력받는 경우 Numpy 배열 연산보다 빠름
    # 길이가 50이하일 경우 cpp geos 엔진을 래핑하여 사용하는 것보다 빠름
    if type(points) == np.ndarray:
        return polyArea_vector(points)
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area

def polyArea_vector(points: np.ndarray):
    # 좌표 입력 데이터가 이미 np array로 취급된다면 일반 배열 연산보다 훨씬 빠르다. 
    # np array 할당에 시간이 가장 많이 소요된다.
    right_shift_points = np.roll(points, 1, axis=0)
    area = np.cross(points, right_shift_points)
    return abs(area.sum()) / 2.0
