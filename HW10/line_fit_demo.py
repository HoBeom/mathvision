import cv2
import numpy as np

window_size = (640, 480)
polygon_close_with_same_point_click = True # for macos


def fitGreenLine(points: np.ndarray):
    # fit y = ax + b
    X = points[:,0]
    Y = points[:,1]
    X1 = np.vstack([X, np.ones(len(X))])
    A = np.linalg.inv(X1 @ X1.T) @ X1
    a, b = A @ Y
    print(f"a:{a}, b:{b}")
    return a, b

def fitRedLine(points: np.ndarray):
    # fit ax + by + c = 0
    X = points[:,0]
    Y = points[:,1]
    A = np.vstack([X, Y, np.ones(len(X))]).T
    a, b, c = np.linalg.svd(A)[-1][-1, :]
    print(f"a:{a}, b:{b}, c:{c}")
    return a, b, c


def on_mouse(event, x, y, buttons, user_param):

    def set_done(points):
        print(f"Completing polygon with {len(points)} points.")
        if len(points) > 2:
            print(f"points:{points}")
            return True
        print("Reject Done polygon with less than 3 points")
        return False
    
    def reset():
        global done, points, current, prev_current, fitdone
        points = []
        current = (x, y)
        prev_current = (0,0)
        done = False
        fitdone = False

    global done, points, current, prev_current, colors
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
                done = set_done(points)
            return
        print("Adding point #%d with position(%d,%d)" % (len(points), x, y))
        points.append([x, y])
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        prev_current = (x, y)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # Double left click means close polygon
        done = set_done(points)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click means done or reset
        if done:
            reset()
        else:
            done = set_done(points)


# mian
if __name__ == '__main__':
    global done, points, current, prev_current, colors, fitdone
    done = False
    fitdone = False
    points = []
    colors = []
    current = (-10,-10)
    prev_current = (0, 0)
    frame = np.ones((window_size[1], window_size[0], 3), dtype=np.uint8) * 255

    cv2.namedWindow("Least Square Demo")
    cv2.setMouseCallback("Least Square Demo", on_mouse)

    while True:
        # display mouse position
        draw_frame = frame.copy()
        # display points
        for i, (point, color) in enumerate(zip(points, colors)):
            x, y = point
            cv2.circle(draw_frame, point, 2, color, 5, -1)  
            cv2.putText(draw_frame, f"{chr(65+i)}({x},{y})", (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 1, cv2.LINE_AA)
        
        if done:
            # display line
            if not fitdone:
                points = np.array(points)
                a1, b1 = fitGreenLine(points)
                x1, y1 = 0, int(b1)
                x2, y2 = window_size[0], int(a1 * window_size[0] + b1)

                a2, b2, c2= fitRedLine(points)
                x3, y3 = 0, int(-c2/b2)
                x4, y4 = window_size[0], int(-(a2 * window_size[0] + c2)/b2)
                a3 = -a2/b2
                b3 = -c2/b2
                fitdone = True
            cv2.line(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(draw_frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(draw_frame, f"Green: y = {a1:.2f}x + {b1:.2f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(draw_frame, f"Red: y = {a3:.2f}x + {b3:.2f}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 1, cv2.LINE_AA)
        else:
            cv2.circle(draw_frame, current, 2, (0, 0, 255), -1)
            cv2.putText(draw_frame, f"({current[0]},{current[1]})", current, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("Least Square Demo", draw_frame)
        if cv2.waitKey(50) == 27:
            print("Escape hit, closing...")
            break

    cv2.destroyWindow("Least Square Demo")