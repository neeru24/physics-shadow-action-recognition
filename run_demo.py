#!/usr/bin/env python3
"""
Quick demo script for Physics-Based Shadow-Depth Action Recognition
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import simplified versions for demo
class DemoShadowDetector:
    """Simplified shadow detector for demo"""
    def detect(self, frame):
        h, w = frame.shape[:2]
        # Mock hand mask (circle in top-left)
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(hand_mask, (100, 100), 50, 255, -1)
        
        # Mock face landmarks
        face_landmarks = np.array([[w//2, h//2]])
        
        # Mock shadow mask
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(shadow_mask, (w//2, h//2), 30, 255, -1)
        
        return hand_mask, face_landmarks, shadow_mask

class DemoPhysicsEngine:
    """Simplified physics engine for demo"""
    def estimate_distance(self, **kwargs):
        # Mock distance that decreases over time
        if not hasattr(self, 'counter'):
            self.counter = 0
        self.counter += 1
        return 10 + 5 * np.sin(self.counter / 10)

def main():
    print("Starting Physics-Based Shadow-Depth Action Recognition Demo")
    print("Press 'q' to quit")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize demo modules
    shadow_detector = DemoShadowDetector()
    physics_engine = DemoPhysicsEngine()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        hand_mask, face_landmarks, shadow_mask = shadow_detector.detect(frame)
        distance = physics_engine.estimate_distance()
        
        # Determine action based on distance
        if distance < 2:
            action = "EATING/SMOKING"
            color = (0, 0, 255)  # Red
        elif distance < 5:
            action = "CLOSE"
            color = (0, 165, 255)  # Orange
        else:
            action = "FAR"
            color = (0, 255, 0)  # Green
        
        # Create visualization
        # Overlay hand mask
        frame[hand_mask > 0] = frame[hand_mask > 0] // 2 + np.array([0, 100, 0])
        
        # Overlay shadow mask
        frame[shadow_mask > 0] = frame[shadow_mask > 0] // 2 + np.array([100, 0, 100])
        
        # Draw text
        cv2.putText(frame, f"Distance: {distance:.1f} cm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Action: {action}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw mock face point
        if len(face_landmarks) > 0:
            fx, fy = face_landmarks[0]
            cv2.circle(frame, (fx, fy), 5, (255, 255, 0), -1)
            cv2.putText(frame, "Face", (fx+10, fy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw mock hand
        cv2.circle(frame, (100, 100), 5, (0, 255, 255), -1)
        cv2.putText(frame, "Hand", (110, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Display
        cv2.imshow("Physics Shadow Demo", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Distance = {distance:.1f} cm, Action = {action}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended")

if __name__ == "__main__":
    main()