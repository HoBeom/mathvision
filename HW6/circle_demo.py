import cv2
import numpy as np

window_size = (640, 480)
polygon_close_with_same_point_click = True # for macos


def fitCircle(points: np.ndarray):
    X = points[:,0]
    Y = points[:,1]
    A = np.array([X, Y, np.ones(len(X))]).T
    b = X**2 + Y**2
    A_peudo = np.linalg.pinv(A)
    a, b, c = A_peudo @ -b
    xc = -a/2
    yc = -b/2
    r = np.sqrt(xc**2 + yc**2 - c)
    return xc, yc, r


def on_mouse(event, x, y, buttons, user_param):

    def close_polygon(points):
        print(f"Completing polygon with {len(points)} points.")
        if len(points) > 2:
            print(f"points:{points}")
            return True
        print("Reject Done polygon with less than 3 points")
        return False
    
    def reset():
        global done, points, current, prev_current
        points = []
        current = (x, y)
        prev_current = (0,0)
        done = False

    global done, points, current, prev_current
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
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # Double left click means close polygon
        done = close_polygon(points)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click means Reset everything
        print("Resetting")
        reset()


# mian
if __name__ == '__main__':
    global done, points, current, prev_current
    done = False
    points = []
    current = (-10,-10)
    prev_current = (0, 0)
    frame = np.ones((window_size[1], window_size[0], 3), dtype=np.uint8) * 255

    cv2.namedWindow("CircleDemo")
    cv2.setMouseCallback("CircleDemo", on_mouse)

    while True:
        draw_frame = frame.copy()
        for i, point in enumerate(points):
            cv2.circle(draw_frame, point,5,(0,200,0),-1)
        
        if done or len(points)>=3:
            # draw fitted circle
            xc, yc, r = fitCircle(np.array(points))
            cv2.circle(draw_frame, (int(xc), int(yc)), int(r), (0, 0, 255), 2)

        cv2.imshow("CircleDemo", draw_frame)
        if cv2.waitKey(50) == 27:
            print("Escape hit, closing...")
            break

cv2.destroyWindow("CircleDemo")