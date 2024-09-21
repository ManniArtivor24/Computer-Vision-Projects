import cv2
import torch
import numpy as np
from time import sleep

# Load YOLOv5 model (trained on COCO dataset to detect 80 classes)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Using yolov5s model
model.classes = None  # Detect all classes available in the model
model.conf = 0.4  # Confidence threshold

# Video capture and other settings
cap = cv2.VideoCapture('video.mp4')
fps_delay = 60  # Frames per second of the video
line_position = 550  # Position of the counting line
offset = 6  # Allowed error between pixels

object_count = 0  # Variable to count the number of objects crossing the line

# Get video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter('output_video.mp4', fourcc, fps_delay, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    time_delay = float(1 / fps_delay)
    sleep(time_delay)

    # Run YOLO object detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Format: [x1, y1, x2, y2, confidence, class]

    # Process each detection
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        class_id = int(cls)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{model.names[class_id]}: {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Check if the object crosses the line
        if y1 < (line_position + offset) and y2 > (line_position - offset):
            object_count += 1
            cv2.line(frame, (25, line_position), (1200, line_position), (0, 127, 255), 3)
            print(f"Object {label} crossed the line. Total Count: {object_count}")

    # Display object count
    cv2.putText(frame, "OBJECT COUNT: " + str(object_count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Write the frame to the output video file
    out.write(frame)

    # Display the frame
    cv2.imshow("Video", frame)

    # Check if 'q' key is pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
out.release()  # Release the VideoWriter object
cv2.destroyAllWindows()
