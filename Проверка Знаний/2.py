import cv2
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0)

lower_green = np.array([40, 100, 100])
upper_green = np.array([70, 255, 255])

def get_ball(hsv_image, lower, upper):
    mask = cv2.inRange(hsv_image, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 10:
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)
            return True, (int(x), int(y), int(radius), mask, (x_rect, y_rect, w_rect, h_rect))
    return False, (-1, -1, -1, np.array([]), (0, 0, 0, 0))

points = []

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    found, (x, y, radius, mask, rect) = get_ball(hsv, lower_green, upper_green)
    if found:
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)

        points.append((x, y, radius))
        print(radius)

    for i in range(1, len(points)):
        _, _, r2 = points[i]
        cv2.line(frame, points[i-1][:2], points[i][:2], (0, 0, 255), r2//2)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        points = []


def draw_with_ball(frame, x, y, radius):
    cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)

capture.release()
cv2.destroyAllWindows()
