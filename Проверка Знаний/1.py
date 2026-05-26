import cv2
import numpy as np
for i in range(10):
    image = cv2.imread(f"0{i}.png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle = contours[0]

    for k, _ in enumerate(contours):
        cv2.drawContours(image, contours, k, (255, 0, 0), 3)

    print(f"area = {np.sum(thresh > 100)}")

    cv2.namedWindow("circle", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("circle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
