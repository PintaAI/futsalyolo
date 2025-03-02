import cv2
import numpy as np
from ultralytics import YOLO

def test_imports():
    print("OpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)
    print("YOLO (Ultralytics) is successfully imported")
    
    # Create a simple test array with NumPy
    arr = np.array([1, 2, 3, 4, 5])
    print("\nNumPy array test:", arr)
    
    # Create a simple blank image with OpenCV
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, "Test", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    print("OpenCV image shape:", img.shape)

if __name__ == "__main__":
    test_imports()
