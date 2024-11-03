import cv2
from ultralytics import YOLO

def run_inference_on_video(video_path, model_path='best.pt'):
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Get the original video width and height
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame while maintaining aspect ratio
        height, width = frame.shape[:2]
        scale = 640 / max(height, width)  # Scale factor
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Perform inference
        results = model(resized_frame)

        # Render results on the frame
        annotated_frame = results[0].plot()  # Add detections to the frame

        # Resize the annotated frame back to the original size for display
        annotated_frame = cv2.resize(annotated_frame, (original_width, original_height))

        # Display the annotated frame
        cv2.imshow('YOLO Object Detection', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Example usage
# model_pt = "license_plate_detector_GPU_NOv6.pt"
model_pt = "yolov8m.pt"
video_path = "cars.mp4"  # Replace with your input video path
run_inference_on_video(video_path, model_pt)
