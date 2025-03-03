import yt_dlp
import os

def download_video(url):
    try:
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best',  # Download best quality
            'outtmpl': 'videos/%(title)s.%(ext)s',  # Output template
            'progress_hooks': [show_progress],  # Progress callback
        }
        
        print("Fetching video information...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            print(f"\nTitle: {info['title']}")
            print(f"Duration: {info['duration']} seconds")
            print(f"Resolution: {info['resolution']}")
            
            # Confirm download
            print("\nStarting download...")
            ydl.download([url])
            
            filename = f"videos/{info['title']}.{info['ext']}"
            print("\nDownload completed!")
            print(f"Saved to: {filename}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

def show_progress(d):
    if d['status'] == 'downloading':
        total_bytes = d.get('total_bytes')
        downloaded_bytes = d.get('downloaded_bytes', 0)
        
        if total_bytes:
            percentage = (downloaded_bytes / total_bytes) * 100
            print(f"Progress: {percentage:.1f}% of {total_bytes/1024/1024:.1f}MB", end='\r')

if __name__ == "__main__":
    print("YouTube Video Downloader")
    print("=" * 20)
    
    while True:
        url = input("\nEnter YouTube URL (or 'q' to quit): ")
        if url.lower() == 'q':
            break
            
        download_video(url)
