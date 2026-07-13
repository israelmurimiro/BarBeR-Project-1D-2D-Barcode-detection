"""
Real-time barcode detection using webcam.

This script uses the trained YOLO model to detect barcodes
from the webcam feed in real-time.
"""

import cv2
from ultralytics import YOLO
import time
from pathlib import Path

# Load the trained model
model_path = "/Users/israelm/Desktop/DATA SCIENCE/runs/detect/train-3/weights/best.pt"
model = YOLO(model_path)

# Class names
class_names = ['1D', '2D']

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("=" * 60)
print("Barcode Detection - Press 'q' to quit")
print("=" * 60)

# FPS calculation
fps_start_time = time.time()
fps_frame_count = 0
fps_display = 0

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLO inference
    results = model(frame, conf=0.25, iou=0.45)

    # Draw detections
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Draw label
                label = f"{class_names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Calculate and display FPS
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        fps_display = fps_frame_count
        fps_frame_count = 0
        fps_start_time = time.time()

    # Display FPS and number of barcodes
    num_detections = len(results[0].boxes) if results[0].boxes else 0
    cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Barcodes: {num_detections}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Barcode Detection", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("✅ Camera closed.")