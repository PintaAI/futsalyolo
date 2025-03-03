import cv2
import numpy as np
import torch
from ultralytics import YOLO
from team_assigner import TeamAssigner

# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLOv8 model and move to GPU
model = YOLO("models/yolov5su.pt")
model.to(device)  # Move model to GPU

cap = cv2.VideoCapture("videos/futsal3.mp4")

# Initialize variables
is_paused = False
team_assigner = TeamAssigner()
next_player_id = 1
initialization_frames = 0
player_detections = {}
color_threshold = 40  # Initial threshold value

while cap.isOpened():
    if not is_paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect only 'person' (index 0) and 'sports ball' (index 32)
        results = model(frame, classes=[0, 32], verbose=False)

        # Process detections
        current_detections = {}
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                label = result.names[int(box.cls[0])]
                conf = box.conf[0].item()

                if label == "person":
                    # Store detection for team assignment
                    bbox = [x1, y1, x2, y2]
                    current_detections[next_player_id] = {"bbox": bbox}
                    next_player_id += 1

                    if initialization_frames < 30:
                        # During initialization, just draw boxes in white
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.putText(frame, f"Player", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                elif label == "sports ball":
                    # Draw ball detection in white
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Handle team assignment and visualization
        if len(current_detections) > 0:
            if initialization_frames == 30:
                # After 30 frames, assign team colors
                team_assigner.assign_team_color(frame, current_detections, color_threshold)
            
            if initialization_frames >= 30:
                # Get and visualize team assignments
                for player_id, detection in current_detections.items():
                    bbox = detection["bbox"]
                    team_id = team_assigner.get_player_team(frame, bbox, player_id, color_threshold)
                    color = team_assigner.team_colors.get(team_id, (255, 255, 255))  # Default to white if team color not assigned yet
                    
                    x1, y1, x2, y2 = bbox
                    # Draw full player bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add team label
                    cv2.putText(frame, f"Team {team_id}", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if initialization_frames <= 30:
                initialization_frames += 1

        # Display color threshold
        cv2.putText(frame, f"Color Threshold: {color_threshold}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame outside the paused check
        cv2.imshow("YOLO Detection (GPU)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):  # Space bar
        is_paused = not is_paused
    elif key == ord("o"):  # Increase threshold
        color_threshold = min(color_threshold + 1, 255)
        initialization_frames = 0  # Reset to reassign teams with new threshold
        team_assigner.team_colors = {}  # Clear existing team colors
        team_assigner.player_team_dict = {}  # Clear existing team assignments
    elif key == ord("p"):  # Decrease threshold
        color_threshold = max(color_threshold - 1, 0)
        initialization_frames = 0  # Reset to reassign teams with new threshold
        team_assigner.team_colors = {}  # Clear existing team colors
        team_assigner.player_team_dict = {}  # Clear existing team assignments

# Clean up
cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()  # Clear CUDA cache
