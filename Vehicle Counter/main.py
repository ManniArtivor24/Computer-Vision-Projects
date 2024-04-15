import cv2
import numpy as np
from time import sleep

min_width = 80      # Minimum width of the rectangle
min_height = 80     # Minimum height of the rectangle
offset = 6          # Allowed error between pixels
line_position = 550  # Position of the counting line
fps_delay = 60      # Frames per second of the video
detections = []     # List to store detected vehicle centers
car_count = 0       # Variable to count the number of cars

def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    center_x = x + x1
    center_y = y + y1
    return center_x, center_y

cap = cv2.VideoCapture('video.mp4')
background_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame1 = cap.read()
    time_delay = float(1 / fps_delay)
    sleep(time_delay)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 5)
    img_subtraction = background_subtractor.apply(blurred)
    dilated = cv2.dilate(img_subtraction, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, line_position), (1200, line_position), (255, 127, 0), 3)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        validate_contour = (w >= min_width) and (h >= min_height)
        if not validate_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = get_center(x, y, w, h)
        detections.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (center_x, center_y) in detections:
            if center_y < (line_position + offset) and center_y > (line_position - offset):
                car_count += 1
                cv2.line(frame1, (25, line_position), (1200, line_position), (0, 127, 255), 3)
                detections.remove((center_x, center_y))
                print("Car detected: " + str(car_count))

    cv2.putText(frame1, "VEHICLE COUNT: " + str(car_count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Original Video", frame1)
    cv2.imshow("Detection", dilated)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()

