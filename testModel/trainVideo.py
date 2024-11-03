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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Render results on the frame
        annotated_frame = results[0].plot()  # Add detections to the frame

        # Display the annotated frame
        cv2.imshow('YOLO Object Detection', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Example usage
model_pt = "license_plate_detector_GPU_NOv6.pt"
video_path = "cars.mp4"  # Replace with your input video path
run_inference_on_video(video_path, model_pt)
