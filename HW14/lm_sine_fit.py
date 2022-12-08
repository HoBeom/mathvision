import cv2
import numpy as np

from lm_solver import LMSolver

window_size = (640, 480)
polygon_close_with_same_point_click = True # for macos

# y = a * sin(b * x + c) + d
func = lambda x, coeffs: coeffs[0] * np.sin(coeffs[1] * x + coeffs[2]) + coeffs[3]
init_coeffs = lambda y: np.array([np.std(y), 0.001, -1.0, np.mean(y)])

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
        current = [x, y]
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Left click means adding a point at current position to the list of points
        if done:
            if not fitdone:
                return
            reset()
        if prev_current == current:
            print("Same point input")
            if polygon_close_with_same_point_click:
                done = set_done(points)
            return
        print("Adding point #%d with position(%d,%d)" % (len(points), x, y))
        points.append([x, y])
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        prev_current = [x, y]
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

    displayname = "fit"
    cv2.namedWindow(displayname)
    cv2.setMouseCallback(displayname, on_mouse, 0)

    while True:
        draw_frame = frame.copy()
        for i, (point, color) in enumerate(zip(points, colors)):
            x, y = point
            cv2.circle(draw_frame, point, 2, (0, 0, 0), 1, -1)
        if done:
            if not fitdone:
                # y = A * sin(B * x + C) + D    
                solver = LMSolver(func)
                x = np.array([p[0] for p in points])
                y = np.array([p[1] for p in points])
                fit_coeffs = init_coeffs(y)
                fit_process = solver.yield_fit(x, y, fit_coeffs)
                for i, coeffs in enumerate(fit_process):
                    fit_frame = draw_frame.copy()
                    cstr = [f"{c:.3f}" for c in coeffs]
                    text = f"iter:{i} y = {cstr[0]} * sin({cstr[1]} * x + {cstr[2]}) + {cstr[3]}"
                    cv2.putText(fit_frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
                    # draw sine wave
                    fit_coeffs = coeffs
                    X = np.linspace(0, window_size[0], window_size[0])
                    Y = func(X, fit_coeffs)
                    sine_points = np.vstack((X, Y)).T
                    sine_points = sine_points.reshape(1, -1, 2).astype(np.int32)
                    cv2.polylines(fit_frame, sine_points, False, (0, 255, 0), 1)
                    cv2.imshow(displayname, fit_frame)
                    if cv2.waitKey(10) == 27:
                        print("Escape hit, fit process break")
                        break
                fitdone = True
            cv2.putText(draw_frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
            # draw sine wave
            cv2.polylines(draw_frame, sine_points, False, (0, 255, 0), 1)

        else:
            cv2.circle(draw_frame, current, 2, (0, 0, 255), -1)
            cv2.putText(draw_frame, f"({current[0]},{current[1]})", current, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.imshow(displayname, draw_frame)
        if cv2.waitKey(50) == 27:
            print("Escape hit, closing...")
            break

    cv2.destroyAllWindows()