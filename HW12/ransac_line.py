import cv2
import numpy as np

window_size = (640, 480)
polygon_close_with_same_point_click = True # for macos


def fitline(points):
    X = points[:,0]
    Y = points[:,1]
    A = np.vstack([X, Y, np.ones(len(X))]).T
    a, b, c = np.linalg.svd(A)[2][-1, :]
    return a, b, c

def RANSAC_line_fit(points, n_sample=3, threshold=20, max_iter=1000):
    """
    RANSAC line fit
    :param points: 2D points
    :parma n_sample: number of points to fit a line
    :param threshold: threshold for inlier
    :param max_iter: max iteration
    :return: a, b, c, support inliers
    """
    X, Y = points[:,0], points[:,1]
    A = np.vstack([X, Y, np.ones(len(X))]).T
    best_inliers = np.zeros_like(X, dtype=np.bool)
    for i in range(max_iter):
        rand_idx = np.random.choice(len(points), n_sample, replace=False)
        model = np.linalg.svd(A[rand_idx])[2][-1, :]
        residual = np.abs(model @ A.T) / np.sqrt(model[0]**2 + model[1]**2)
        inliers = residual < threshold
        if inliers.sum() > best_inliers.sum():
            print(f"iter:{i}, inliers:{len(inliers)}")
            best_inliers = inliers
    final_model = np.linalg.svd(A[best_inliers])[2][-1, :]
    residual = np.abs(final_model @ A.T) / np.sqrt(model[0]**2 + model[1]**2)
    return *final_model, points[residual < threshold]

def get_success_rate(num_points, n_support_set, n_samples, max_iter):
    """
    Compute the success rate of RANSAC
    :param num_points: number of points
    :param n_support_set: number of inliers
    :param n_samples: number of points to fit a line
    :param max_iter: max iteration
    :return: success rate
    """
    from math import comb
    nCp = comb(num_points, n_samples)
    eCp = comb(n_support_set, n_samples)
    fail_prob = 1 - eCp / nCp
    return 1 - fail_prob ** max_iter

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
        current = [x, y]
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

    cv2.namedWindow("RANSAC Line Demo")
    cv2.setMouseCallback("RANSAC Line Demo", on_mouse)

    while True:
        # display mouse position
        draw_frame = frame.copy()
        # display points
        for i, (point, color) in enumerate(zip(points, colors)):
            x, y = point
            cv2.circle(draw_frame, point, 2, (0, 0, 0), 1, -1)
            # cv2.circle(draw_frame, point, 2, color, 5, -1)  
            # cv2.putText(draw_frame, f"{chr(65+i)}({x},{y})", (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 1, cv2.LINE_AA)
        
        if done:
            # display line
            n_sample, threshold, max_iter = 3, 20, 1000

            if not fitdone:
                a, b, c, inliers = RANSAC_line_fit(np.array(points), n_sample, threshold, max_iter)
                x1, y1 = 0, int(-c/b)
                x2, y2 = window_size[0], int(-(a * window_size[0] + c)/b)
                lsa, lsb, lsc = fitline(np.array(points))
                x3, y3 = 0, int(-lsc/lsb)
                x4, y4 = window_size[0], int(-(lsa * window_size[0] + lsc)/lsb)
                success_rate = get_success_rate(len(points), len(inliers), n_sample, max_iter)
                fitdone = True
            cv2.putText(draw_frame, f"N_Sample:{n_sample}, Threshold:{threshold}, Max_Iter:{max_iter}, Success_Rate:{success_rate:.4f}", (10,15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 1, cv2.LINE_AA)
            cv2.line(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(draw_frame, f"RANSAC(RED): {a:.6f}x + {b:.6f}y + {c:.2f} = 0", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 1, cv2.LINE_AA)
            cv2.line(draw_frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.putText(draw_frame, f"LS(GREEN): {lsa:.6f}x + {lsb:.6f}y + {lsc:.2f} = 0", (10, 45), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 1, cv2.LINE_AA)
            # display inliers 
            inliers = np.array(inliers)
            for i, point in enumerate(inliers):
                x, y = point
                cv2.circle(draw_frame, point, 1, (255, 0, 0), 5, -1)
        else:
            cv2.circle(draw_frame, current, 2, (0, 0, 255), -1)
            cv2.putText(draw_frame, f"({current[0]},{current[1]})", current, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("RANSAC Line Demo", draw_frame)
        if cv2.waitKey(50) == 27:
            print("Escape hit, closing...")
            break

    cv2.destroyWindow("RANSAC Line Demo")