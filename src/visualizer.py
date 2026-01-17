import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.colors = {
            'touching': (0, 0, 255),      # Red
            'eating': (0, 165, 255),      # Orange
            'drinking': (0, 255, 255),    # Yellow
            'close': (0, 255, 0),         # Green
            'far': (255, 0, 0)            # Blue
        }
    
    def draw_overlay(self, frame, distance, action, face_landmarks, hand_landmarks, shadow_mask):
        """Draw all visualizations on frame"""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw face landmarks
        if face_landmarks:
            for landmark in face_landmarks.landmark[::10]:  # Draw every 10th point
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(output, (x, y), 2, (0, 255, 255), -1)
        
        # Draw hand landmarks
        if hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(output, (x, y), 3, (255, 255, 0), -1)
        
        # Draw shadow mask with transparency
        if shadow_mask is not None:
            shadow_colored = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
            shadow_colored[:, :, 0] = 255  # Blue shadow
            output = cv2.addWeighted(output, 0.7, shadow_colored, 0.3, 0)
        
        # Determine color based on action
        if "TOUCHING" in action or "EATING" in action:
            color = self.colors['touching']
        elif "DRINKING" in action:
            color = self.colors['drinking']
        elif "CLOSE" in action:
            color = self.colors['close']
        else:
            color = self.colors['far']
        
        # Draw distance text
        cv2.putText(output, f"Distance: {distance:.1f} cm", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw action text
        cv2.putText(output, f"Action: {action}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw intensity matrix in corner
        matrix = self._create_intensity_matrix(distance)
        matrix_resized = cv2.resize(matrix, (100, 100))
        output[10:110, w-110:w-10] = matrix_resized
        
        # Draw matrix label
        cv2.putText(output, "Intensity", (w-110, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    def _create_intensity_matrix(self, distance):
        """Create intensity heatmap visualization"""
        size = 100
        matrix = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Create gradient based on distance
        center_intensity = max(0, min(255, int(255 * (1 - distance/50))))
        
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i-size//2)**2 + (j-size//2)**2)
                intensity = center_intensity * np.exp(-dist/(size/4))
                matrix[i, j] = [0, intensity, 255-intensity]  # Blue to yellow
        
        return matrix
    
    def draw_no_detection(self, frame):
        """Draw when no detection"""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        cv2.putText(output, "No face/hand detected", (w//2-150, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(output, "Show face and hand to camera", (w//2-200, h//2+50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output