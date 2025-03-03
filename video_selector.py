import os
import cv2

class VideoSelector:
    def __init__(self, videos_dir="videos"):
        self.videos_dir = videos_dir
        self.video_list = self._get_video_list()
        self.current_video_index = 0
        self.current_video = None
        self.current_video_name = None

    def _get_video_list(self):
        """Get list of video files from the videos directory."""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        videos = [f for f in os.listdir(self.videos_dir) 
                 if os.path.isfile(os.path.join(self.videos_dir, f)) 
                 and f.lower().endswith(video_extensions)]
        return sorted(videos)

    def get_current_video(self):
        """Get current video capture object."""
        if not self.video_list:
            return None
        
        if self.current_video is None:
            video_path = os.path.join(self.videos_dir, self.video_list[self.current_video_index])
            self.current_video = cv2.VideoCapture(video_path)
            self.current_video_name = self.video_list[self.current_video_index]
        
        return self.current_video

    def next_video(self):
        """Switch to next video in the list."""
        if not self.video_list:
            return None

        # Release current video if it exists
        if self.current_video is not None:
            self.current_video.release()

        # Update index
        self.current_video_index = (self.current_video_index + 1) % len(self.video_list)
        
        # Open new video
        video_path = os.path.join(self.videos_dir, self.video_list[self.current_video_index])
        self.current_video = cv2.VideoCapture(video_path)
        self.current_video_name = self.video_list[self.current_video_index]
        
        return self.current_video

    def previous_video(self):
        """Switch to previous video in the list."""
        if not self.video_list:
            return None

        # Release current video if it exists
        if self.current_video is not None:
            self.current_video.release()

        # Update index
        self.current_video_index = (self.current_video_index - 1) % len(self.video_list)
        
        # Open new video
        video_path = os.path.join(self.videos_dir, self.video_list[self.current_video_index])
        self.current_video = cv2.VideoCapture(video_path)
        self.current_video_name = self.video_list[self.current_video_index]
        
        return self.current_video

    def get_current_video_name(self):
        """Get name of current video file."""
        if self.current_video_name:
            return self.current_video_name
        return None

    def release(self):
        """Release current video capture object."""
        if self.current_video is not None:
            self.current_video.release()
            self.current_video = None
