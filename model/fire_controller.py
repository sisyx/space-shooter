import mediapipe as mp
import numpy as np

class FireController:
    def __init__(self):
        self.initialized = True

    def decide(self, points):
        """Predict whether to fire based on hand points"""
        
        angle = self.calculate_angle_3d(points[0], points[1], points[2])
        
        return bool(60 < angle < 165)

    def calculate_angle_3d(self, pt1, pt2, pt3):
        """
        Calculate angle between two 3D lines that share a common point.
        
        Args:
            pt1: Start of line 1 (x, y, z) - can be MediaPipe landmark or tuple/list
            pt2: End of line 1 / Start of line 2 (x, y, z) - the shared vertex
            pt3: End of line 2 (x, y, z)
        
        Returns:
            float: Angle in degrees between the two lines
        """

        if hasattr(pt1, 'x'): 
            p1 = np.array([pt1.x, pt1.y, pt1.z])
            p2 = np.array([pt2.x, pt2.y, pt2.z])
            p3 = np.array([pt3.x, pt3.y, pt3.z])
        else:  # Regular coordinates
            p1 = np.array(pt1)
            p2 = np.array(pt2)
            p3 = np.array(pt3)
        
        # Create vectors from the shared point (pt2)
        vector1 = p1 - p2  # From pt2 to pt1
        vector2 = p3 - p2  # From pt2 to pt3
        
        # Calculate magnitudes
        mag1 = np.linalg.norm(vector1)
        mag2 = np.linalg.norm(vector2)
        
        # Handle edge case: zero-length vectors
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        # Calculate dot product
        dot_product = np.dot(vector1, vector2)
        
        # Calculate cosine of angle
        cos_angle = dot_product / (mag1 * mag2)
        
        # Clamp to valid range [-1, 1] to handle floating point errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Calculate angle in radians, then convert to degrees
        angle_radians = np.arccos(cos_angle)
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees

if __name__ == "__main__":
    fire_controller = FireController()