"""
Shadow detection and analysis module
Detects shadow regions cast by hand on face
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class ShadowDetector:
    def __init__(self):
        """Initialize shadow detector with face and hand models"""
        # Initialize MediaPipe models
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Shadow detection parameters
        self.shadow_threshold = 0.3  # Intensity drop threshold
        self.min_shadow_contour_area = 50
        
        logger.info("Shadow Detector initialized")
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect hand, face landmarks, and shadow regions
        
        Returns:
        --------
        hand_mask : np.ndarray
            Binary mask of hand region
        face_landmarks : np.ndarray
            Facial landmarks (68 points, normalized)
        shadow_mask : np.ndarray
            Binary mask of shadow region on face
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        face_landmarks = self._detect_face_landmarks(rgb_frame)
        
        # Detect hand
        hand_mask = self._detect_hand(rgb_frame)
        
        # Detect shadow region
        shadow_mask = self._detect_shadow_region(frame, face_landmarks, hand_mask)
        
        return hand_mask, face_landmarks, shadow_mask
    
    def _detect_face_landmarks(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Detect facial landmarks using MediaPipe"""
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return np.array([])
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array (normalized coordinates)
        h, w, _ = rgb_frame.shape
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        
        return np.array(landmarks)
    
    def _detect_hand(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Detect hand and create binary mask"""
        results = self.hands.process(rgb_frame)
        
        h, w, _ = rgb_frame.shape
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        
        if not results.multi_hand_landmarks:
            return hand_mask
        
        # Get hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Create convex hull around hand
        points = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        if len(points) > 0:
            hull = cv2.convexHull(points)
            cv2.fillPoly(hand_mask, [hull], 255)
        
        return hand_mask
    
    def _detect_shadow_region(self, 
                             frame: np.ndarray, 
                             face_landmarks: np.ndarray,
                             hand_mask: np.ndarray) -> np.ndarray:
        """
        Detect shadow region cast by hand on face
        
        Algorithm:
        1. Extract face region
        2. Calculate expected illumination pattern
        3. Detect intensity drops beyond expected
        4. Filter by proximity to hand and face geometry
        """
        if len(face_landmarks) == 0:
            return np.zeros_like(frame[:, :, 0])
        
        # Create face mask using convex hull of landmarks
        h, w = frame.shape[:2]
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Use subset of landmarks for face outline
        face_outline_indices = list(range(1, 16)) + [33, 133, 362, 263, 33]
        face_points = face_landmarks[face_outline_indices].astype(np.int32)
        
        cv2.fillPoly(face_mask, [face_points], 255)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Calculate expected illumination (simplified - would use light model)
        # For now, use Gaussian blur to estimate local illumination
        expected_illumination = cv2.GaussianBlur(gray, (31, 31), 0)
        
        # Calculate intensity deviation
        intensity_diff = expected_illumination - gray
        
        # Threshold to find dark regions (shadows)
        shadow_candidates = (intensity_diff > self.shadow_threshold).astype(np.uint8) * 255
        
        # Apply face mask
        shadow_candidates = cv2.bitwise_and(shadow_candidates, face_mask)
        
        # Remove small regions
        shadow_mask = self._filter_shadow_regions(shadow_candidates)
        
        # Ensure shadow is near hand (optional constraint)
        if np.sum(hand_mask) > 0:
            # Calculate distance from hand
            hand_dist = cv2.distanceTransform(255 - hand_mask, cv2.DIST_L2, 3)
            
            # Only keep shadows close to hand
            max_hand_distance = 100  # pixels
            shadow_mask[hand_dist > max_hand_distance] = 0
        
        return shadow_mask
    
    def _filter_shadow_regions(self, mask: np.ndarray) -> np.ndarray:
        """Filter shadow regions by size and shape"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= self.min_shadow_contour_area:
                # Additional shape filtering could be added here
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask
    
    def analyze_shadow(self, 
                      frame: np.ndarray, 
                      shadow_mask: np.ndarray,
                      face_landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Analyze shadow properties
        
        Returns:
        --------
        shadow_area : float
            Area of shadow region (pixels)
        shadow_intensity : float
            Average intensity drop in shadow region (0-1)
        """
        if np.sum(shadow_mask) == 0:
            return 0.0, 0.0
        
        # Calculate shadow area
        shadow_area = np.sum(shadow_mask > 0)
        
        # Calculate average intensity in shadow region
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_normalized = gray.astype(np.float32) / 255.0
        
        # Calculate average intensity in shadow vs non-shadow face regions
        face_intensity = gray_normalized[shadow_mask > 0]
        
        if len(face_intensity) == 0:
            shadow_intensity = 0.0
        else:
            # Calculate expected intensity for face region (simplified)
            # In practice, would use skin tone model
            expected_skin_intensity = 0.6  # Typical skin tone
            
            # Shadow intensity is ratio of actual to expected
            actual_intensity = np.mean(face_intensity)
            shadow_intensity = actual_intensity / expected_skin_intensity
        
        return shadow_area, np.clip(shadow_intensity, 0.0, 1.0)
    
    def calculate_shadow_sharpness(self, shadow_mask: np.ndarray) -> float:
        """
        Calculate shadow edge sharpness using gradient analysis
        """
        if np.sum(shadow_mask) == 0:
            return 0.0
        
        # Apply Sobel filter for edge detection
        sobel_x = cv2.Sobel(shadow_mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(shadow_mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize and calculate sharpness
        max_gradient = np.max(gradient_mag)
        if max_gradient > 0:
            sharpness = np.mean(gradient_mag) / max_gradient
        else:
            sharpness = 0.0
        
        return np.clip(sharpness, 0.0, 1.0)
    
    def estimate_shadow_volume(self, 
                              shadow_mask: np.ndarray,
                              face_landmarks: np.ndarray) -> float:
        """
        Estimate 3D volume of shadow (simplified)
        
        Note: This is a placeholder for more sophisticated 3D reconstruction
        """
        # Simplified: volume ≈ area × depth_factor
        shadow_area = np.sum(shadow_mask > 0)
        
        # Estimate depth factor from face curvature
        if len(face_landmarks) >= 68:
            # Use nose to cheek distance as depth proxy
            nose = face_landmarks[30]
            left_cheek = face_landmarks[2]
            right_cheek = face_landmarks[14]
            
            # Calculate approximate face width
            face_width = np.linalg.norm(left_cheek - right_cheek)
            depth_factor = face_width / 100.0  # Simplified scaling
        else:
            depth_factor = 1.0
        
        shadow_volume = shadow_area * depth_factor
        
        return shadow_volume