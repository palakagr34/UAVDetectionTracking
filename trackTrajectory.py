import cv2
import numpy as np
import os
from filterpy.kalman import KalmanFilter
from detectDrones import detect_drones
from filterpy.common import Q_discrete_white_noise


# Initialize Kalman filter
kf = KalmanFilter(dim_x=4, dim_z=2)


# Set initial state 
detections_folder = "./detections"
files = os.listdir(detections_folder)
image_path = os.path.join(detections_folder, files[0])

first_image = cv2.imread(image_path)
height, width, _ = first_image.shape

initial_state = [width/2, height/2, 0, 0]  # [x, y, dx, dy]  => center of frame
kf.x = np.array(initial_state)

# Initialize previous detection coordinates
prev_detection_x, prev_detection_y = None, None

# Define the transition matrix
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

# Define the measurement function
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])

# Process and measurement noise covariance
kf.P *= 1000.
kf.R = 5
kf.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.13)

# Path to detections folder
detections_folder = "./detections"

output_video_path = "path_to_output_video.mp4"
frame_width = width  
frame_height = height  
print("width: ", width)
print("height: ", height)
output_video = cv2.VideoWriter(output_video_path,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               30, (frame_width, frame_height))

run = 0
# Loop through the detections folder
for filename in os.listdir(detections_folder):
    run+=1
    print(f"Run: {run}")
    if filename.endswith(".jpg"):
        frame = cv2.imread(os.path.join(detections_folder, filename))
        
        detected_boxes = detect_drones(frame)
        print(f"Detected boxes: {detected_boxes}")


        if len(detected_boxes) > 0:
            max_confidence_idx = np.argmax(detected_boxes[:, -1])  # Assuming last column is the confidence score
            chosen_box = detected_boxes[max_confidence_idx]

            x1, y1, x2, y2 = chosen_box  # Assuming the box is in format [x1, y1, x2, y2]
            measurement = [(x1 + x2) / 2, (y1 + y2) / 2]  # Using center of the chosen box as measurement


            prediction = kf.predict()
            print(f"Prediction: {prediction}")  

            if prediction is not None: 
                predicted_x, predicted_y = prediction[:2]

                kf.update(measurement)

                frame = cv2.circle(frame, (int(predicted_x), int(predicted_y)), 5, (0, 255, 0), -1)

                if prev_detection_x is not None and prev_detection_y is not None:
                    frame = cv2.line(frame, (int(prev_detection_x), int(prev_detection_y)),
                                (int(predicted_x), int(predicted_y)), (0, 255, 0), 2)

                prev_detection_x, prev_detection_y = predicted_x, predicted_y

        output_video.write(frame)
        
output_video.release()
