import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    
    Parameters:
    a, b, c (list or np.array): Coordinates of three points 
                                [x, y, z] or [x, y]
    
    Returns:
    float: Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate cosine of the angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Convert to angle in degrees
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert from radians to degrees
    return np.degrees(angle)


def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ) as pose:
        
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and find pose landmarks
            results = pose.process(image)
            
            # Convert back to BGR for display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Check if pose is detected
            if results.pose_landmarks:
                # Extract specific landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for shoulder, elbow, and wrist
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                    # 0
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z
                ]
                
                elbow = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                    # 0
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].z
                ]
                
                wrist = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                    # 0
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].z
                ]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Display angle on the image
                cv2.putText(
                    image, 
                    f'Angle: {round(angle, 2)} degrees', 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
            
            # Display the image
            cv2.imshow('Violin Pose Angle', image)
            
            # Exit condition
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()