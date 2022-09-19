#!/usr/bin/python3
import os.path as osp

import cv2
import numpy as np
from utils import classifyHomography

MIN_MATCH_NUM = 4
WEB_CAM_MODE = False

def proc_video(video_capture: cv2.VideoCapture, model: np.ndarray):
    model_gray = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)

    # local feature method
    fd = cv2.ORB_create()

    # Find the keypoints with ORB
    # obj_keypts, descsobj_descriptors = fd.detectAndCompute(model_gray, None)
    
    # detect model key points
    obj_keypts = fd.detect(model_gray, None)
    if len(obj_keypts) < MIN_MATCH_NUM:
        return

    # compute descriptor of the model key points
    obj_descriptors = fd.compute(model_gray, obj_keypts)

    # descriptor matcher
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    bf = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING, crossCheck=False)

    while True:
        # Capture frame
        ret, frame = video_capture.read()
        if not ret:
            break
        draw_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        img_keypts = fd.detect(gray, None)
        if len(img_keypts) < MIN_MATCH_NUM:
            continue

        # compute descriptor of the model key points
        img_descriptors = fd.compute(gray, img_keypts)
        
        matches = bf.match(obj_descriptors[1], img_descriptors[1])
        if len(matches) < MIN_MATCH_NUM:
            continue

        dmatches = sorted(matches, key = lambda x:x.distance)

        # extract the matched keypoints
        src_pts  = np.float32([obj_keypts[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
        dst_pts  = np.float32([img_keypts[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

        # find homography matrix and do perspective transform
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        if len(H) != 0:
            h, w = model.shape[:2]
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, H)

            # draw found regions
            draw_frame = cv2.polylines(draw_frame, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)

            ## classify homography
            homo_type = classifyHomography(pts.reshape(4,2), dst.reshape(4,2))
            
            cv2.putText(draw_frame, f"{homo_type}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # draw match lines
        draw_frame = cv2.drawMatches(model, obj_keypts, draw_frame, img_keypts, dmatches[:20],None,flags=2)

        cv2.imshow("Match", draw_frame)
        ch = cv2.waitKey(1)
        if ch == 27: return
        elif ch == 32:
            while True:
                stop = cv2.waitKey(10)
                if stop == 27: return
                if stop == 32: break

def web_cam_process(video_capture: cv2.VideoCapture):
    
    global drawing, L_point, R_point
    drawing = False
    L_point = [0, 0]
    R_point = [0, 0]

    def mouse_drawing(event, x, y, flags, params):
        global drawing, L_point, R_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            L_point = [x, y]
            R_point = [x+1, y+1]
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing is True:
                R_point = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            R_point = [x, y]

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_drawing)

    while True:
        _, frame = cap.read()
        # copy frame
        if frame is None:
            raise ValueError("No Video Frame")
        draw_frame = frame.copy()

        cv2.rectangle(draw_frame, L_point, R_point, (0, 255, 0), 2)
        cv2.imshow("Frame", draw_frame)

        key = cv2.waitKey(1)
        if key == 32:
            cv2.destroyAllWindows()
            break
    
    # point condition check
    if L_point[0] > R_point[0]:
        L_point[0], R_point[0] = R_point[0], L_point[0]
    if L_point[1] > R_point[1]:
        L_point[1], R_point[1] = R_point[1], L_point[1]
    
    model = frame[L_point[1]:R_point[1], L_point[0]:R_point[0]]
    proc_video(video_capture, model)


# main 
if __name__ == "__main__":
    
    if WEB_CAM_MODE:
        video_capture = cv2.VideoCapture(0)
        web_cam_process(video_capture)
        exit(0)

    DATA_DIR = "data/"
    model_paths = ["blais.jpg", "mousepad.bmp"]
    video_paths = ["blais.mp4", "mousepad.mp4"]

    for model_path, video_path in zip(model_paths, video_paths):
        # load object model
        try:
            model = cv2.imread(osp.join(DATA_DIR, model_path))
            video_capture = cv2.VideoCapture(osp.join(DATA_DIR, video_path))
        except FileNotFoundError:
            print(f"File not found {model_path} or {video_path}")
            continue

        # process video
        proc_video(video_capture, model)

        cv2.destroyAllWindows()
