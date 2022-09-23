import cv2
import numpy as np
from utils import classifyHomography, polyArea

check_homography = False
window_size = (640, 480)
polygon_close_with_same_point_click = True # for macos

def on_mouse(event, x, y, buttons, user_param):

    def close_polygon(points):
        print(f"Completing polygon with {len(points)} points.")
        if len(points) > 2:
            print(f"points:{points}")
            return True
        print("Reject Done polygon with less than 3 points")
        return False
    
    def reset():
        global done, points, current, prev_current, frame, homography_type
        points = []
        current = (x, y)
        prev_current = (0,0)
        done = False
        homography_type = None


    global done, points, current, prev_current, frame
    if event == cv2.EVENT_MOUSEMOVE:
        if done:
            return
        current = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Left click means adding a point at current position to the list of points
        if done:
            reset()
        if not check_homography and prev_current == current:
            print("Same point input")
            if polygon_close_with_same_point_click:
                done = close_polygon(points)
            return
        print("Adding point #%d with position(%d,%d)" % (len(points), x, y))
        points.append((x, y))
        prev_current = (x, y)
        if check_homography and len(points) == 4:
            done = close_polygon(points)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # Double left click means close polygon
        if check_homography:
            print("Double click to close polygon")
            done = close_polygon(points)
        else:
            print("Complete polygon with 4 points")
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click means Reset everything
        print("Resetting")
        reset()


# mian
if __name__ == '__main__':
    global done, points, current, prev_current, frame, homography_type
    done = False
    points = []
    current = (-10,-10)
    prev_current = (0,0)
    frame = np.ones((window_size[1], window_size[0], 3), dtype=np.uint8) * 255
    homography_type = None

    if check_homography:
        size = window_size[1] // 10
        orignal_rect = np.array([[0, 0], [0, size], [size, size], [size, 0]], dtype=np.float32)
    
    cv2.namedWindow("PolygonDemo")
    cv2.setMouseCallback("PolygonDemo", on_mouse)

    while True:
        # This is our drawing loop, we just continuously draw new images
        # and show them in the named window
        draw_frame = frame.copy()
        if len(points) == 0:
            text = " Check Homography" if check_homography else " Polygon Area"
            cv2.putText(draw_frame, "Input data points (double click: finish)" + text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 148, 0), 1, cv2.LINE_AA)
        else:
            if(current != prev_current):
                cv2.line(draw_frame, (points[-1][0], points[-1][1]), current, (0,0,255))
        
        for i, point in enumerate(points):
            cv2.circle(draw_frame, point,5,(0,200,0),-1)
            cv2.putText(draw_frame, chr(65+i), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        

        if not done and (len(points) > 1):
            cv2.circle(draw_frame, current, 5,(0,200,200),-1)
            cv2.polylines(draw_frame, [np.array(points)], False, (255,0,0), 1)
        
        if done:
            cv2.polylines(draw_frame, [np.array(points)], True, (255,0,0), 1)
            if check_homography:
                cv2.putText(draw_frame, "Homography completed", (10, window_size[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 148, 0), 1, cv2.LINE_AA)
                cv2.rectangle(draw_frame, (0, 0), (size, size), (0, 0, 255), 1)
                # draw line from points to orignal_rect
                for i in range(4):
                    cv2.line(draw_frame, (int(points[i][0]), int(points[i][1])), 
                            (int(orignal_rect[i][0]), int(orignal_rect[i][1])), (200,200,0))
                # check homography
                if homography_type is None:
                    homography_type = classifyHomography(np.array(points, dtype=np.float32), orignal_rect)
                cv2.putText(draw_frame, f"{homography_type}", (10, window_size[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 148, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(draw_frame, "Polygon completed", (10, window_size[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 148, 0), 1, cv2.LINE_AA)
                area = polyArea(np.array(points))
                cv2.putText(draw_frame, f"Area: {area}", (10, window_size[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 148, 0), 1, cv2.LINE_AA)            
            
        cv2.imshow("PolygonDemo", draw_frame)
        if cv2.waitKey(50) == 27:
            print("Escape hit, closing...")
            break

cv2.destroyWindow("PolygonDemo")