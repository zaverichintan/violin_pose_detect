import cv2
import numpy as np
import onnxruntime as ort
import os
import time 
from collections import deque
import argparse
import sys

def arg_parser():
    """
    Creates argument parser.
    """

    parser = argparse.ArgumentParser()

    # Input File
    parser.add_argument('path',
                        type=str,
                        help='Input file path, pass 0 to get webcam')
    return parser

class PoseDetector:
    def __init__(self, model_path, trajectory_length=50):
        """
        Initialize ONNX pose detection model
        
        :param model_path: Path to the ONNX model file
        """
        # Initialize ONNX Runtime inference session
        self.session = ort.InferenceSession(model_path)
        
        # Input and output names (may vary depending on the specific model)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Model input configuration
        self.input_height = 192
        self.input_width = 192

        # Keypoint labels (MoveNet standard order)
        self.keypoint_labels = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Violin-specific points of interest
        self.violin_relevant_points = [
            'left_shoulder', 'right_shoulder', 
            'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist'
        ]
    
        # Initialize trajectory storage
        self.right_wrist_trajectory = deque(maxlen=trajectory_length)
        self.right_elbow_trajectory = deque(maxlen=trajectory_length)
    
        
    def preprocess_frame(self, frame):
        """
        Preprocess the input frame for the ONNX model
        
        :param frame: Input image
        :return: Preprocessed input tensor
        """
        # Resize frame to model's expected input size
        input_frame = cv2.resize(frame, (self.input_width, self.input_height))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        input_frame = input_frame.astype(np.float32)

        # Add batch dimension
        input_frame = np.expand_dims(input_frame, axis=0)
        return input_frame
    
    def detect_pose(self, frame):
        """
        Detect pose in the input frame
        
        :param frame: Input image
        :return: Detected pose keypoints
        """
        # Preprocess the frame
        input_tensor = self.preprocess_frame(frame)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name], 
            {self.input_name: input_tensor}
        )
        
        # Process and return pose keypoints
        keypoints = self.process_outputs(outputs[0], frame)
        # Update trajectories if wrist is detected
        if 'right_wrist' in keypoints and keypoints['right_wrist']['confidence'] > 0.3:
            self.right_wrist_trajectory.append(keypoints['right_wrist']['coordinates'])
        if 'right_elbow' in keypoints and keypoints['right_elbow']['confidence'] > 0.3:
            self.right_elbow_trajectory.append(keypoints['right_elbow']['coordinates'])
            
        return keypoints
    
    def process_outputs(self, raw_outputs, original_frame):
        """
        Process model outputs to extract pose keypoints
        
        :param raw_outputs: Raw model outputs
        :param original_frame: Original input frame
        :return: Processed pose keypoints
        """
        height, width = original_frame.shape[:2]
        processed_keypoints = {}
        
        keypoint_data = raw_outputs.squeeze()
        for i, keypoint in enumerate(keypoint_data):
            # Each keypoint has 3 values: x, y, confidence
            y, x, confidence = keypoint
            
            # Scale coordinates back to original frame size
            scaled_x = int(x * width)
            scaled_y = int(y * height)
            
            # Store keypoint with its scaled coordinates and confidence
            label = (self.keypoint_labels[i] 
                     if i < len(self.keypoint_labels) 
                     else f'keypoint_{i}')
            
            processed_keypoints[label] = {
                'coordinates': (scaled_x, scaled_y),
                'confidence': float(confidence)
            }
        
        
        return processed_keypoints
    
    def draw_trajectory(self, frame):
        """
        Draw the motion path of the right wrist on the frame
        
        :param frame: Input frame to draw on
        :return: Frame with trajectory drawn
        """
        if len(self.right_wrist_trajectory) > 1:
            # Convert trajectory to numpy array for easier processing
            points = np.array(self.right_wrist_trajectory, np.int32)
            
            # Draw the trajectory line
            cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 255), thickness=2)
            
            # Draw circles at each point
            for i, point in enumerate(self.right_wrist_trajectory):
                # Make the most recent point more visible
                if i == len(self.right_wrist_trajectory) - 1:
                    cv2.circle(frame, point, 5, (0, 255, 255), -1)
                else:
                    cv2.circle(frame, point, 2, (255, 0, 255), -1)
        
        return frame
    
    def calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points.
        
        Parameters:
        a, b, c (list or np.array): Coordinates of three points 
                                    [x, y, z] or [x, y]
        
        Returns:
        float: Angle in degrees
        """
        a = np.array(a['coordinates'])
        b = np.array(b['coordinates'])
        c = np.array(c['coordinates'])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate cosine of the angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Convert to angle in degrees
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        # Convert from radians to degrees
        return np.degrees(angle)

    def analyze_violin_pose(self, keypoints, confidence_threshold=0.5):
        """
        Analyze violin playing pose, specifically checking elbow position
        
        :param keypoints: Detected keypoints
        :param confidence_threshold: Minimum confidence for keypoint consideration
        :return: Analysis results
        """
        analysis_results = {
            'elbow_above_shoulder': False,
            'elbow_shoulder_details': None,
            'elbow_shoulder_wrist_angle': 0
        }
        
        # Check if relevant keypoints exist and have sufficient confidence
        try:
            right_wrist = keypoints.get('right_wrist', None)
            right_elbow = keypoints.get('right_elbow', None)
            right_shoulder = keypoints.get('right_shoulder', None)
            
            # Verify keypoints exist and meet confidence threshold
            if (right_elbow and right_shoulder and
                right_elbow['confidence'] > confidence_threshold and
                right_shoulder['confidence'] > confidence_threshold):
                
                # Compare y-coordinates (lower y means higher in image) as the origin is on top left
                elbow_y = right_elbow['coordinates'][1]
                shoulder_y = right_shoulder['coordinates'][1]
                
                # Check if elbow is above shoulder
                analysis_results['elbow_above_shoulder'] = elbow_y < shoulder_y
                
                # Store detailed information
                analysis_results['elbow_shoulder_details'] = {
                    'elbow_y': elbow_y,
                    'shoulder_y': shoulder_y,
                    'vertical_distance': shoulder_y - elbow_y,
                    
                }
                # store the angle
                analysis_results['elbow_shoulder_wrist_angle'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        except Exception as e:
            print(f"Error in pose analysis: {e}")
        
        return analysis_results
    
def detect_violin_player_pose(video_path, model_path):
    """
    Perform pose detection on a video file
    
    :param video_path: Path to the input video file
    :param model_path: Path to the ONNX pose detection model
    """
    # Validate input paths
    if not os.path.exists(video_path) and video_path != '0':
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")
    
    # Initialize pose detector
    pose_detector = PoseDetector(model_path)
    
    # Open video capture
    if video_path == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Optional: Set up video writer to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('task_a_output_pose_detection.mp4', fourcc, fps, (width, height))
    
    elbow_above_shoulder_frames = 0
    total_frames = 0
    start_time = time.time()
    
    # For storing wrist movement data for analysis
    wrist_movement_data = []

    while cap.isOpened():
        # Read frame from video
        success, frame = cap.read()
        if not success:
            break        
        # Detect pose
        try:
            keypoints = pose_detector.detect_pose(frame)
            
            # Analyze pose
            pose_analysis = pose_detector.analyze_violin_pose(keypoints)
            
            # Update tracking
            total_frames += 1
            if pose_analysis['elbow_above_shoulder']:
                elbow_above_shoulder_frames += 1
            
            # Store wrist movement data if available
            if 'right_wrist' in keypoints and keypoints['right_wrist']['confidence'] > 0.3:
                wrist_movement_data.append({
                    'frame': total_frames,
                    'position': keypoints['right_wrist']['coordinates']
                })

            frame = pose_detector.draw_trajectory(frame)

            # Visualize keypoints
            for label, kp in keypoints.items():
                # Only draw keypoints with confidence above a threshold
                if kp['confidence'] > 0:
            
                    # Optional: Add label
                    if label in pose_detector.violin_relevant_points:
                        if label == 'right_elbow' and pose_analysis['elbow_above_shoulder']:
                            cv2.circle(frame, kp['coordinates'], 5, (0, 0, 255), -1)
                        else:
                            cv2.circle(frame, kp['coordinates'], 5, (0, 255, 0), -1)
            
                        cv2.putText(frame, 
                                f"{label} {kp['confidence']:.2f})", 
                                (kp['coordinates'][0]+10, kp['coordinates'][1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    

            # Indicate elbow position analysis
            analysis_text = (
                "Elbow Above Shoulder" if pose_analysis['elbow_above_shoulder'] 
                else "Elbow Below Shoulder"
            )
            cv2.putText(frame, analysis_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 0, 255) if pose_analysis['elbow_above_shoulder'] else (0, 255, 0), 
                        2)
            # Display angle on the image
            cv2.putText(
                frame, 
                f'Angle: {round(pose_analysis['elbow_shoulder_wrist_angle'], 2)} degrees', 
                (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            if total_frames % 30:
                fps = total_frames // (time.time() - start_time)
            
            cv2.putText(frame, f"FPS: ({fps})", (10, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255, 0, 5),
                        2)
            
            # Write frame to output video
            out.write(frame)
            
            # Display the frame (optional)
            cv2.imshow('Violin Player Pose Detection', frame)
            
            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Pose detection completed. Output saved to 'output_pose_detection.mp4'")

     # Print overall analysis
    print(f"Total Frames: {total_frames}")
    print(f"Frames with Elbow Above Shoulder: {elbow_above_shoulder_frames}")
    print(f"Percentage: {(elbow_above_shoulder_frames/total_frames)*100:.2f}%")

def main():
    # Paths for video and ONNX model
    subparser = arg_parser()
    args = subparser.parse_args(sys.argv[1:])
    print('Processing file -> {}'.format(args.path))

    VIDEO_PATH = args.path
    
    MODEL_PATH = '/Users/chintanzaveri/Downloads/neptune/model/model_float32.onnx'
    
    # Run pose detection
    detect_violin_player_pose(VIDEO_PATH, MODEL_PATH)

if __name__ == "__main__":
    main()