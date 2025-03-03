import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("futsal.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect only 'person' (index 0) and 'sports ball' (index 32)
    results = model(frame, classes=[0, 32])
    persons = []  # To store data for detected persons

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            conf = box.conf[0].item()

            if label == "person":
                # Determine the jersey area (upper half)
                y_mid = y1 + (y2 - y1) // 2
                roi = frame[y1:y_mid, x1:x2]
                # Compute mean HSV color; if ROI is empty, default to [0,0,0]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) if roi.size else np.zeros((1, 1, 3))
                mean_color = cv2.mean(hsv_roi)[:3]
                # Save bbox, mean color, and y_mid for drawing the ROI later
                persons.append({'bbox': (x1, y1, x2, y2), 'color': mean_color, 'y_mid': y_mid})
            elif label == "sports ball":
                # Draw sports ball detections in white
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Cluster jersey colors with K-Means if two or more persons are detected
    if len(persons) >= 2:
        colors = np.array([p['color'] for p in persons])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(colors)
        teams = kmeans.labels_
    else:
        teams = [0] * len(persons)

    # Define two team colors (BGR format)
    team_colors = [(255, 0, 0), (0, 255, 0)]  # Blue and Green

    # Draw bounding boxes for persons and their jersey ROI
    for i, person in enumerate(persons):
        x1, y1, x2, y2 = person['bbox']
        y_mid = person['y_mid']
        team = teams[i]
        # Draw main bounding box for the person with team color
        cv2.rectangle(frame, (x1, y1), (x2, y2), team_colors[team], 2)
        cv2.putText(frame, f"Team {team+1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_colors[team], 2)
        # Draw the jersey area (upper half) rectangle in yellow
        cv2.rectangle(frame, (x1, y1), (x2, y_mid), (0, 255, 255), 2)

    cv2.imshow("YOLO + KMeans", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()