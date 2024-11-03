import cv2
from ultralytics import YOLO

def load_and_predict(image_path, model_path, save_path):
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Load the image
    image = cv2.imread(image_path)

    # Make predictions
    results = model.predict(image, conf=0.6)  # Adjust confidence threshold if needed

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]  # x1, y1, x2, y2 coordinates
            confidence = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = model.names[cls]  # Get class label from class index

            # Draw bounding box on the image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image with predictions
    cv2.imwrite(save_path, image)
    print(f"Predicted image saved to: {save_path}")

if __name__ == '__main__':
    # Path to the image you want to test
    image = "car2"
    decoder = "jpg"
    image_path = image+"."+decoder  # Replace with your image path
    # Path to your trained model weights
    model_path = 'license_plate_detector_GPU_NOv6.pt'  # Replace with your best model path
    # Path to save the predicted image
    save_path = image + "pred." + decoder  # Replace with your desired save path

    load_and_predict(image_path, model_path, save_path)
