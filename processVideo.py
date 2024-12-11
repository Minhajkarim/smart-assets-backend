#!/usr/bin/env python
import cv2
import sys
import os
import json
from YOLO_Pred import YOLO_Pred

# Get the video path from arguments
video_path = sys.argv[1]
output_dir = 'videos'
os.makedirs(output_dir, exist_ok=True)

# Initialize YOLO model
model_path = 'models/best.onnx'
data_yaml = 'models/data.yaml'
yolo_model = YOLO_Pred(model_path, data_yaml)

# Load the video
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

# Output video path
output_video_path = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '-output.mp4'))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame using YOLO model
    processed_frame = yolo_model.predictions(frame)

    # Write the processed frame to the output video
    out.write(processed_frame)

    # Update progress and send it to stdout for the backend
    frame_num += 1
    progress = (frame_num / frame_count) * 100
    progress_update = {
        'progress': int(progress),
        'message': f'Processing frame {frame_num}/{frame_count}'
    }
    print(json.dumps(progress_update))  # Send progress update

# Release resources
cap.release()
out.release()

# Final output path
output_video_info = {
    'output_video': output_video_path
}
print(json.dumps(output_video_info))  # Send the final output video path to the backend
