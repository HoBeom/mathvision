import cv2
import numpy as np

window_size = (640, 480)
polygon_close_with_same_point_click = True # for macos

def fitEllipse(points: np.ndarray):
    X = points[:,0]
    Y = points[:,1]
    D = np.array([X**2, X*Y, Y**2, X, Y, np.ones(len(X))]).T
    U, S, V = np.linalg.svd(D)
    print(f"U:{U}")
    print(f"S:{S}")
    a, b, c, d, e, f = V[-1]
    print(f"a:{a}, b:{b}, c:{c}, d:{d}, e:{e}, f:{f}")
    # Extract Ellipse Parameters
    cx = (2*c*d - b*e) / (b**2 - 4*a*c)
    cy = (2*a*e - b*d) / (b**2 - 4*a*c)
    cu = a*cx**2 + b*cx*cy + c* cy*cy - f
    # angle
    theta = 0.5 * np.arctan2(b, a-c)
    theta = theta * 180 / np.pi
    # major and minor axis
    # when points len under 6 if not perfect ellipse will cause error
    # so we shoud add abs before sqrt (not necessary but prevent error just in case)
    major_axis = np.sqrt(abs(2*cu / (a + c + np.sqrt((a-c)**2 + b**2))))
    minor_axis = np.sqrt(abs(2*cu / (a + c - np.sqrt((a-c)**2 + b**2))))
    # vaild check
    print(f"cx:{cx}, cy:{cy}, major_axis:{major_axis}, minor_axis:{minor_axis}, theta:{theta}")
    return cx, cy, major_axis, minor_axis, theta
    

def on_mouse(event, x, y, buttons, user_param):

    def close_polygon(points):
        print(f"Completing polygon with {len(points)} points.")
        if len(points) > 2:
            print(f"points:{points}")
            return True
        print("Reject Done polygon with less than 3 points")
        return False
    
    def reset():
        global done, points, current, prev_current, cal_ellipse
        cal_ellipse = True
        points = []
        current = (x, y)
        prev_current = (0,0)
        done = False

    global done, points, current, prev_current, cal_ellipse
    if event == cv2.EVENT_MOUSEMOVE:
        if done:
            return
        current = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Left click means adding a point at current position to the list of points
        if done:
            reset()
        if prev_current == current:
            print("Same point input")
            if polygon_close_with_same_point_click:
                done = close_polygon(points)
            return
        print("Adding point #%d with position(%d,%d)" % (len(points), x, y))
        points.append([x, y])
        prev_current = (x, y)
        cal_ellipse = True
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # Double left click means close polygon
        done = close_polygon(points)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click means Reset everything
        print("Resetting")
        reset()


# mian
if __name__ == '__main__':
    global done, points, current, prev_current, cal_ellipse
    done = False
    points = []
    current = (-10,-10)
    prev_current = (0, 0)
    cal_ellipse = True

    frame = np.ones((window_size[1], window_size[0], 3), dtype=np.uint8) * 255
    cv2.namedWindow("EllipseDemo")
    cv2.setMouseCallback("EllipseDemo", on_mouse)
        
    while True:
        draw_frame = frame.copy()
        for i, point in enumerate(points):
            cv2.circle(draw_frame, point,5,(0,200,0),-1)
        
        if done or len(points) >= 3:
            # draw fitted circle
            # cv2.ellipse(draw_frame, cv2.fitEllipse(np.array(points)), (0, 0, 255), 2)
            if cal_ellipse:
                cx, cy, ma, mi, angle = fitEllipse(np.array(points))
                if len(points) >= 5:
                    print(cv2.fitEllipse(np.array(points))) # compare with cv2 fitEllipse
                    (cv2cx, cv2cy), (cv2ma, cv2mi), cv2angle = cv2.fitEllipse(np.array(points))
                cal_ellipse = False
            cv2.ellipse(draw_frame, (int(cx), int(cy)), (int(ma), int(mi)), angle, 0, 360, (255, 0, 255), 1)

        cv2.imshow("EllipseDemo", draw_frame)
        if cv2.waitKey(50) == 27:
            print("Escape hit, closing...")
            break

cv2.destroyWindow("EllipseDemo")