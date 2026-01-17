class ActionClassifier:
    def __init__(self):
        self.thresholds = {
            'eating': 5.0,      # Less than 5cm
            'drinking': 8.0,    # 5-8cm
            'close': 15.0,      # 8-15cm
            'far': 100.0        # More than 15cm
        }
    
    def classify(self, distance):
        """Classify action based on distance"""
        if distance < 2.0:
            return "TOUCHING FACE"
        elif distance < self.thresholds['eating']:
            return "EATING/SMOKING"
        elif distance < self.thresholds['drinking']:
            return "DRINKING"
        elif distance < self.thresholds['close']:
            return "CLOSE PROXIMITY"
        else:
            return "FAR"