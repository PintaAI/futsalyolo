import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("futsal2.mp4")

# Add state variable for pause
is_paused = False

while cap.isOpened():
    if not is_paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect only 'person' (index 0) and 'sports ball' (index 32)
        results = model(frame, classes=[0, 32], verbose=False)

        # Process detections - utilize numpy for batch operations
        for result in results:
            boxes = result.boxes
            # Convert all coordinates to numpy array at once
            coords = boxes.xyxy.cpu().numpy().astype(np.int32)
            classes = boxes.cls.cpu().numpy().astype(np.int32)
            confs = boxes.conf.cpu().numpy()
            
            # Process each detection using numpy arrays
            for i, (coord, cls_id, conf) in enumerate(zip(coords, classes, confs)):
                x1, y1, x2, y2 = coord
                label = result.names[cls_id]
                
                if label == "person":
                    # Draw person detection in green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif label == "sports ball":
                    # Draw sports ball detections in white
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show frame outside the paused check
        cv2.imshow("YOLO Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):  # Space bar
        is_paused = not is_paused

cap.release()
cv2.destroyAllWindows()
