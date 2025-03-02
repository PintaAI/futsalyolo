import cv2
import torch
import numpy as np

def load_model():
    """Load the YOLOv5 model for player and ball detection."""
    print("[INFO] Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [0, 32]  # Detect "person" (0) and "sports ball" (32) classes
    return model

def process_frame(frame, model, roi):
    """Process the frame by detecting players and balls within the defined ROI."""
    height, width = frame.shape[:2]
    y_start, y_end = int(height * roi[0]), int(height * roi[1])
    
    # Extract ROI from frame
    roi_frame = frame[y_start:y_end, :]
    
    # Run YOLOv5 inference
    results = model(roi_frame)
    
    # Extract detection results
    detections = results.xyxy[0].cpu().numpy()
    
    # Adjust Y-coordinates to match the original frame
    if detections.size > 0:
        detections[:, [1, 3]] += y_start  
    
    return detections, (y_start, y_end)

def draw_arrow(frame, x, y, color, size=40):
    """Draw downward-pointing arrow for ball detection."""
    # Calculate arrow points
    top = (int(x), int(y))
    bottom = (int(x), int(y + size))
    left = (int(x - size//3), int(y + size - size//3))
    right = (int(x + size//3), int(y + size - size//3))
    
    # Draw arrow shaft
    cv2.line(frame, top, bottom, color, 2)
    # Draw arrow head
    cv2.line(frame, bottom, left, color, 2)
    cv2.line(frame, bottom, right, color, 2)

def draw_detections(frame, detections, roi_coords):
    """Draw bounding boxes, arrows, and count players detected in the frame."""
    y_start, y_end = roi_coords
    player_count = 0
    ball_count = 0

    # Draw ROI boundary lines
    cv2.line(frame, (0, y_start), (frame.shape[1], y_start), (0, 0, 255), 2)
    cv2.line(frame, (0, y_end), (frame.shape[1], y_end), (0, 0, 255), 2)
    
    for x1, y1, x2, y2, conf, cls in detections:
        cls = int(cls)
        if cls == 0:  # Person detection
            player_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Player {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif cls == 32:  # Ball detection
            ball_count += 1
            ball_center_x = (x1 + x2) / 2
            ball_y = y1  # Arrow starts from top of ball detection
            draw_arrow(frame, ball_center_x, ball_y, (0, 255, 255))  # Yellow arrow for ball
            cv2.putText(frame, f'Ball {conf:.2f}', (int(ball_center_x), int(ball_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Display counts
    cv2.putText(frame, f'Players: {player_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Balls: {ball_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

def main():
    """Main function to process the futsal video."""
    model = load_model()  
    roi = (0.3, 0.9)  # Define ROI: Ignore top 30% and bottom 10% of the frame
    frame_skip = 2  # Process every nth frame (adjust this value to control performance)
    frame_count = 0
    last_detections = None
    last_roi_coords = None

    print("[INFO] Opening video file...")
    video = cv2.VideoCapture('futsal.mp4')

    if not video.isOpened():
        print("[ERROR] Could not open video file.")
        return
    
    print("[INFO] Processing video... Press 'q' to quit.")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Only process every nth frame
        if frame_count % frame_skip == 0:
            # Detect players and balls
            detections, roi_coords = process_frame(frame, model, roi)
            last_detections = detections
            last_roi_coords = roi_coords
        else:
            # Use last detections for skipped frames
            detections = last_detections if last_detections is not None else np.array([])
            roi_coords = last_roi_coords if last_roi_coords is not None else (0, 0)

        frame_count += 1

        # Draw results
        processed_frame = draw_detections(frame.copy(), detections, roi_coords)

        # Display frame
        cv2.imshow('Player and Ball Detection', processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()
    print("[INFO] Processing complete.")

if __name__ == "__main__":
    main()
