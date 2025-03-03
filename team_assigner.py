import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.color_history = defaultdict(list)  # Store color history for each player
        self.history_size = 5  # Number of frames to keep track of
    
    def process_jersey_color(self, image):
        if image.size == 0:  # Check if image is empty
            return None, image
            
        # Get average RGB color directly
        try:
            avg_color = np.mean(image, axis=(0,1))
            if np.isnan(avg_color).any():  # Check for NaN values
                return None, image
        except:
            return None, image
            
        return avg_color, image
    
    def get_jersey_color(self, frame, bbox):
        # Get the original bbox coordinates
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1
        
        # Calculate the narrower width (40% margin from each side)
        width_margin = int(width * 0.4)  # Reduced from 0.6 to capture more of jersey
        narrow_x1 = x1 + width_margin
        narrow_x2 = x2 - width_margin
        
        # Ensure minimum width of 2 pixels
        if narrow_x2 - narrow_x1 < 2:
            narrow_x1 = x1 + width // 3
            narrow_x2 = x2 - width // 3
        
        # Calculate ROI height (focus on chest area with larger region)
        height_start = int(height * 0.25)  # Start higher (was 0.3)
        height_end = int(height * 0.5)     # End lower (was 0.4)
        narrow_y1 = y1 + height_start
        narrow_y2 = y1 + height_end
        
        # Ensure minimum height of 2 pixels
        if narrow_y2 - narrow_y1 < 2:
            narrow_y1 = y1 + height // 3
            narrow_y2 = y2 - height // 3
        
        # Get the narrower image
        try:
            image = frame[narrow_y1:narrow_y2, narrow_x1:narrow_x2]
            jersey_color, jersey_img = self.process_jersey_color(image)
            return jersey_color, jersey_img
        except:
            return None, None

    def assign_team_color(self, frame, player_detections, threshold=15):  # Lowered threshold
        jersey_colors = []
        
        # Collect colors from all players
        for player_id, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            result = self.get_jersey_color(frame, bbox)
            if result is None or result[0] is None:
                continue
                
            jersey_color, _ = result
            if not np.all(jersey_color < threshold):  # Skip extremely dark colors
                jersey_colors.append(jersey_color)
        
        if len(jersey_colors) < 2:
            return
            
        # Ensure we have valid data
        jersey_colors = np.array(jersey_colors)
        if np.isnan(jersey_colors).any():
            return
            
        # Cluster jersey colors in RGB space
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(jersey_colors)
        self.kmeans = kmeans
        
        # Store cluster centers as team colors
        for team_id, center in enumerate(kmeans.cluster_centers_):
            self.team_colors[team_id + 1] = tuple(map(int, center))
            
    def get_average_team(self, player_id):
        """Get the most frequent team assignment from history"""
        if not self.color_history[player_id]:
            return None
        teams = [team for team in self.color_history[player_id]]
        return max(set(teams), key=teams.count)

    def get_player_team(self, frame, player_bbox, player_id, threshold=15):  # Lowered threshold
        # Return cached team assignment if available
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        result = self.get_jersey_color(frame, player_bbox)
        if result is None or result[0] is None:
            # Check history before defaulting
            avg_team = self.get_average_team(player_id)
            return avg_team if avg_team is not None else 1
            
        jersey_color, _ = result
        
        # Handle case when detection failed, check history first
        if np.all(jersey_color < threshold):
            avg_team = self.get_average_team(player_id)
            return avg_team if avg_team is not None else 1
            
        team_id = self.kmeans.predict(jersey_color.reshape(1, -1))[0] + 1
        
        # Special case handling
        if player_id == 91:
            team_id = 1
            
        # Update color history
        self.color_history[player_id].append(team_id)
        if len(self.color_history[player_id]) > self.history_size:
            self.color_history[player_id].pop(0)
            
        # Use the most common team assignment from recent history
        avg_team = self.get_average_team(player_id)
        if avg_team is not None:
            team_id = avg_team
            
        self.player_team_dict[player_id] = team_id
        return team_id
