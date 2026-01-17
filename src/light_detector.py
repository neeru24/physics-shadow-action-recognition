import cv2
import numpy as np

class LightDetector:
    def __init__(self):
        self.prev_frame = None
    
    def estimate(self, frame):
        """Simple light direction estimation"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients (simplified light direction)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Average gradient direction (simplified light direction)
        avg_grad_x = np.mean(grad_x)
        avg_grad_y = np.mean(grad_y)
        
        # Normalize to unit vector
        magnitude = np.sqrt(avg_grad_x**2 + avg_grad_y**2)
        if magnitude > 0:
            light_dir = np.array([avg_grad_x/magnitude, avg_grad_y/magnitude, 0.5])
        else:
            light_dir = np.array([0, 0, 1])  # Default overhead light
        
        # Normalize
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        return light_dir