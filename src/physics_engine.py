import numpy as np

class PhysicsEngine:
    def __init__(self):
        self.calibration_constant = 50.0  # Will be calibrated
        self.prev_distance = None
    
    def estimate(self, shadow_area, hand_area, light_direction, frame):
        """Estimate distance using shadow physics"""
        if shadow_area == 0 or hand_area == 0:
            return 100.0  # Default far distance
        
        # Core physics equation: distance ∝ sqrt(shadow_area / hand_area)
        area_ratio = shadow_area / max(hand_area, 1)
        
        # Apply inverse square law approximation
        distance = self.calibration_constant * np.sqrt(area_ratio)
        
        # Consider light direction (simplified)
        light_factor = 1.0 + 0.5 * light_direction[2]  # Z component affects distance
        distance *= light_factor
        
        # Apply temporal smoothing
        if self.prev_distance is not None:
            alpha = 0.3  # Smoothing factor
            distance = alpha * distance + (1 - alpha) * self.prev_distance
        
        self.prev_distance = distance
        
        # Ensure reasonable bounds
        distance = np.clip(distance, 1.0, 100.0)
        
        return distance
    
    def get_intensity_matrix(self):
        """Generate intensity heatmap for visualization"""
        # Create a simple 8x8 intensity matrix
        matrix = np.ones((8, 8))
        
        # Simulate intensity drop in center
        for i in range(8):
            for j in range(8):
                dist = np.sqrt((i-4)**2 + (j-4)**2)
                matrix[i, j] = 1.0 - 0.8 * np.exp(-dist/2)
        
        return matrix