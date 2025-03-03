import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Load YOLOv8 model
model = YOLO("yolov5s.pt")
cap = cv2.VideoCapture("futsal.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect only 'person' (index 0) and 'sports ball' (index 32)
    results = model(frame, classes=[0, 32], verbose=False)
    persons = []  # To store data for detected persons

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            conf = box.conf[0].item()

            if label == "person":
                # Calculate ROI with padding
                height = y2 - y1
                width = x2 - x1
                
                # Define ROI boundaries (20% from top, 35% height, 20% padding from sides)
                roi_top = y1 + int(height * 0.2)  # Moved up from 30% to 20%
                roi_bottom = roi_top + int(height * 0.35)  # Reduced height to 35%
                roi_left = x1 + int(width * 0.2)
                roi_right = x2 - int(width * 0.2)
                
                roi = frame[roi_top:roi_bottom, roi_left:roi_right]
                
                # Convert to HSV and create mask to ignore black/very dark pixels
                if roi.size:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    # Create mask to ignore very dark pixels (low V value in HSV)
                    mask = hsv_roi[:,:,2] > 30  # Value threshold
                    if np.any(mask):  # If we have any valid pixels
                        mean_color = cv2.mean(hsv_roi, mask=mask.astype(np.uint8))[:3]
                    else:
                        mean_color = np.zeros(3)
                else:
                    mean_color = np.zeros(3)
                
                # Save bbox, mean color, and ROI coordinates for drawing later
                persons.append({
                    'bbox': (x1, y1, x2, y2),
                    'color': mean_color,
                    'roi': (roi_left, roi_top, roi_right, roi_bottom)
                })
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
        roi_left, roi_top, roi_right, roi_bottom = person['roi']
        team = teams[i]
        # Draw main bounding box for the person with team color
        cv2.rectangle(frame, (x1, y1), (x2, y2), team_colors[team], 2)
        cv2.putText(frame, f"Team {team+1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_colors[team], 2)
        # Draw the jersey ROI area in yellow
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 255), 2)

    cv2.imshow("YOLO + KMeans", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
