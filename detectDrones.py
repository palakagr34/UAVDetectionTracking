import os
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path

# Pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def detect_drones(frame):
    # OpenCV frame to a PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    image_tensor = F.to_tensor(pil_image).unsqueeze(0)

    # predictions
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract bounding boxes, labels, and scores
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Filter based on 50% confidence
    threshold = 0.5
    indices = scores > threshold
    boxes = boxes[indices]
    labels = labels[indices]

    return boxes

def process_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frameNum = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video
    print(f"Total frames: {total_frames}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        ret, frame = cap.read()
        frameNum +=1 
        if not ret:
            break

        
        # Detect drones in the frame
        drone_boxes = detect_drones(frame)
        if len(drone_boxes) > 0:
            frame_path = os.path.join(output_folder, f"{Path(video_path).stem}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        progress = (frameNum / total_frames) * 100
        print(f"Progress: {progress:.2f}%")



    # Release video capture object
    cap.release()

if __name__ == "__main__":
    video_dir = "./Videos"

    # Process each video in the directory
    for video_file in os.listdir(video_dir):
        print("Starting new file... ... ... ")
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_file)
            output_folder = "detections" 
            process_video(video_path, output_folder)
