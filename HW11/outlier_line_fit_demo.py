import cv2
import numpy as np
# algorithm from https://github.com/SeolMuah/mathvision/blob/master/HW11_Least_Square/q2_main.py
window_size = (640, 480)
polygon_close_with_same_point_click = True # for macos

def fitCauchylines(points: np.ndarray, iters=10):
    iter_lines = []
    X = points[:,0]
    Y = points[:,1]
    X1 = np.vstack([X, np.ones(len(X))])
    A = np.linalg.inv(X1 @ X1.T) @ X1
    p = A @ Y
    iter_lines.append(p)
    for i in range(iters):
        r = np.abs(Y - X1.T @ p)
        W = np.diag(1 / (r/1.3998+1))
        p = np.linalg.inv(X1 @ W @ X1.T) @ X1 @ W @ Y
        iter_lines.append(p)
    return iter_lines

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

    global done, points, current, prev_current, colors, fitdone
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
        fitdone = False
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
        
        if not fitdone:
            if len(points) > 2:
                fit_points = np.array(points)
                fit_lines = fitCauchylines(fit_points, iters=10)
                fitdone = True
        if fitdone:
            linecolor = np.linspace(0, 255, len(fit_lines))
            green2red = np.vstack([np.zeros(len(fit_lines)), 255 - linecolor, linecolor]).T
            for i, lines in enumerate(fit_lines):
                a, b = lines
                x1 = 0
                y1 = int(a * x1 + b)
                x2 = window_size[0]
                y2 = int(a * x2 + b)
                cv2.line(draw_frame, (x1, y1), (x2, y2), green2red[i], 2)
                cv2.putText(draw_frame, f"y = {a:.2f}x + {b:.2f}", (10, 30 + i * 20), cv2.FONT_HERSHEY_PLAIN, 1.0, green2red[i], 1, cv2.LINE_AA)
        if done:
            # display line
            pass
        else:
            cv2.circle(draw_frame, current, 2, (0, 0, 255), -1)
            cv2.putText(draw_frame, f"({current[0]},{current[1]})", current, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("Least Square Demo", draw_frame)
        if cv2.waitKey(50) == 27:
            print("Escape hit, closing...")
            break

    cv2.destroyWindow("Least Square Demo")