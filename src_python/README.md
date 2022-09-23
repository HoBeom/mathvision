# MathVision - Python template

python >= 3.6

## 스켈레톤 코드 실행 방법
```
$ python polygon_demo.py
or
$ cd python_template && python feature_matching.py
```

## 옵션
1. polygon_demo.py  
```
check_homography = False # True로 변경 시 Homography 검사 모드로 동작
window_size = (640, 480) # 윈도우 사이즈 변경 (가로, 세로)
polygon_close_with_same_point_click = True # for macos (맥북의 경우 더블클릭 불가능)
```
2. feature_matching.py
```
MIN_MATCH_NUM = 4 # 최소 매칭 피쳐 개수
WEB_CAM_MODE = False # Ture로 변경 시 웹캠으로 크롭하여 추적하는 모드로 동작
```

## 알고리즘 구현 필요 부분
utils.py
1. 다각형 면적 계산
   - def polyArea()
2. Homography 변환 종류 검사
   - def classifyHomography()
