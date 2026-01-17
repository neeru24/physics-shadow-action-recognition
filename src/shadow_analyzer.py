import cv2
import numpy as np
import mediapipe as mp

class ShadowAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hand_area = 0
    
    def detect(self, frame):
        """Detect face and hand landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face
        face_results = self.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
        else:
            face_landmarks = None
        
        # Detect hand
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            # Calculate hand area (simplified)
            self._calculate_hand_area(hand_landmarks, frame.shape)
        else:
            hand_landmarks = None
        
        return face_landmarks, hand_landmarks
    
    def _calculate_hand_area(self, landmarks, frame_shape):
        """Calculate approximate hand area"""
        h, w = frame_shape[:2]
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        
        if len(points) > 0:
            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)
            self.hand_area = cv2.contourArea(hull)
        else:
            self.hand_area = 100  # Default
    
    def detect_shadow(self, frame, face_landmarks):
        """Detect shadow on face"""
        if face_landmarks is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8), 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create face mask
        h, w = gray.shape
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get face outline points
        face_points = []
        indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 
                  361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176]
        
        for idx in indices:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                face_points.append([x, y])
        
        if len(face_points) > 2:
            face_points = np.array(face_points, dtype=np.int32)
            cv2.fillPoly(face_mask, [face_points], 255)
        
        # Detect dark regions on face
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = blurred - gray
        
        # Threshold for shadow
        _, shadow_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        # Apply face mask
        shadow_mask = cv2.bitwise_and(shadow_mask, face_mask)
        
        # Calculate shadow area
        shadow_area = np.sum(shadow_mask > 0)
        
        return shadow_mask, shadow_area
    
    def get_hand_area(self):
        return self.hand_area