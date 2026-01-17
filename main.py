#!/usr/bin/env python3
"""
Main application for Physics-Based Shadow-Depth Action Recognition
"""

import cv2
import numpy as np
import argparse
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from src folder
from src.light_detector import LightDetector
from src.shadow_analyzer import ShadowAnalyzer
from src.physics_engine import PhysicsEngine
from src.action_classifier import ActionClassifier
from src.visualizer import Visualizer

class ShadowDepthSystem:
    def __init__(self):
        print("Initializing Physics-Based Shadow-Depth Action Recognition System...")
        
        # Initialize modules
        self.light_detector = LightDetector()
        self.shadow_analyzer = ShadowAnalyzer()
        self.physics_engine = PhysicsEngine()
        self.action_classifier = ActionClassifier()
        self.visualizer = Visualizer()
        
        # State variables
        self.current_distance = None
        self.current_action = "No Action"
        self.frame_count = 0
        
        print("System initialized successfully!")
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        
        # Step 1: Detect face and hand
        face_landmarks, hand_landmarks = self.shadow_analyzer.detect(frame)
        
        if face_landmarks is None or hand_landmarks is None:
            return self.visualizer.draw_no_detection(frame)
        
        # Step 2: Detect light source
        light_direction = self.light_detector.estimate(frame)
        
        # Step 3: Detect shadow
        shadow_mask, shadow_area = self.shadow_analyzer.detect_shadow(frame, face_landmarks)
        
        # Step 4: Estimate distance using physics
        self.current_distance = self.physics_engine.estimate(
            shadow_area=shadow_area,
            hand_area=self.shadow_analyzer.get_hand_area(),
            light_direction=light_direction,
            frame=frame
        )
        
        # Step 5: Classify action
        self.current_action = self.action_classifier.classify(self.current_distance)
        
        # Step 6: Visualize
        output_frame = self.visualizer.draw_overlay(
            frame=frame,
            distance=self.current_distance,
            action=self.current_action,
            face_landmarks=face_landmarks,
            hand_landmarks=hand_landmarks,
            shadow_mask=shadow_mask
        )
        
        return output_frame
    
    def run_realtime(self, source=0):
        """Run on webcam"""
        cap = cv2.VideoCapture(source)
        
        print("Press 'q' to quit, 'c' to calibrate")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            output = self.process_frame(frame)
            
            # Display
            cv2.imshow("Shadow-Depth Action Recognition", output)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibrate()
            
            # Print status every 30 frames
            if self.frame_count % 30 == 0 and self.current_distance:
                print(f"Frame {self.frame_count}: Distance = {self.current_distance:.1f}cm, Action = {self.current_action}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def calibrate(self):
        """Simple calibration"""
        print("Calibration: Place hand at known distance (e.g., 10cm) from face")
        input("Press Enter when ready...")
        print("Calibration complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="0 for webcam or video file path")
    parser.add_argument("--mode", type=str, default="realtime", choices=["realtime", "video"])
    
    args = parser.parse_args()
    
    # Initialize system
    system = ShadowDepthSystem()
    
    # Run
    if args.mode == "realtime":
        source = 0 if args.source == "0" else args.source
        system.run_realtime(source)
    else:
        print("Video mode not implemented yet")

if __name__ == "__main__":
    main()