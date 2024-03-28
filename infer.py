import argparse
from ultralytics import YOLO
import json
import os
import cv2

def detect_fire_smoke_image(folder_path, model_weight):
    # Load a pretrained YOLOv8n model
    model = YOLO(model_weight)

    # List all files in the folder
    image_files = os.listdir(folder_path)

    # Iterate through each image file
    for image_file in image_files:
        # Construct the full path to the image
        image_path = os.path.join(folder_path, image_file)
        
        # Perform detection on the image
        results = model.predict(image_path, save=True, line_width=1, conf=0.47, imgsz=[720,1080])
        
        fire_detected = False
        smoke_detected = False
        
        # Process detection results for each image
        for result in results:
            if not result:
                continue
            detection = json.loads(result.tojson())
            for obj in detection:
                if obj['name'] == 'Fire':
                    fire_detected = True
                elif obj['name'] == 'Smoke':
                    smoke_detected = True
        
        # Check if both fire and smoke are detected simultaneously in the image
        if fire_detected and smoke_detected:
            print(f"Alert: Both fire and smoke detected in {image_file}!")

def detect_fire_smoke_video(video_path, model_weight):
    # Load a pretrained YOLOv8n model
    model = YOLO(model_weight)

    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection on the frame
        results = model.predict(frame, save=True, line_width=1, conf=0.47, imgsz=[720,1080])
        
        fire_detected = False
        smoke_detected = False
        
        # Process detection results for the frame
        for result in results:
            if not result:
                continue
            detection = json.loads(result.tojson())
            for obj in detection:
                if obj['name'] == 'Fire':
                    fire_detected = True
                elif obj['name'] == 'Smoke':
                    smoke_detected = True
        
        # Check if both fire and smoke are detected simultaneously in the frame
        if fire_detected and smoke_detected:
            print("Alert: Both fire and smoke detected in the video!")
            break

    # Release video capture
    cap.release()
    
    # Check if both fire and smoke are detected simultaneously in the video
    if fire_detected and smoke_detected:
        print("Alert: Both fire and smoke detected in the video!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect fire and smoke in images and/or video.")
    parser.add_argument("-f", "--folder_path", help="Path to the folder containing images")
    parser.add_argument("-v", "--video_path", help="Path to the video file")
    parser.add_argument("-w", "--model_weight", required=True, help="Path to the model weight file")

    args = parser.parse_args()

    if args.folder_path and not args.video_path:
        detect_fire_smoke_image(args.folder_path, args.model_weight)
    elif args.video_path and not args.folder_path:
        detect_fire_smoke_video(args.video_path, args.model_weight)
    elif args.folder_path and args.video_path:
        detect_fire_smoke_image(args.folder_path, args.model_weight)
        detect_fire_smoke_video(args.video_path, args.model_weight)
    else:
        print("Please provide either the folder path or the video path.")
