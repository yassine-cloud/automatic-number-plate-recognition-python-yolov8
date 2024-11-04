import cv2
from ultralytics import YOLO
import string
import easyocr

reader = easyocr.Reader(['en'], gpu=True)
# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

# def license_complies_format(text):
#     """
#     Check if the license plate text complies with the required format.

#     Args:
#         text (str): License plate text.

#     Returns:
#         bool: True if the license plate complies with the format, False otherwise.
#     """
#     if len(text) != 7:
#         return False

#     if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
#        (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
#        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
#        (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
#        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
#        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
#        (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
#         return True
#     else:
#         return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    
    # Ensure text length is at least 7 characters to avoid index errors
    if len(text) < 7:
        return text  # Return the unformatted text if it's too short
    
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        # if license_complies_format(text):
         # Ensure text length before formatting
        if len(text) > 0:
            return format_license(text), score

    return None, None

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
            x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]  # Convert tensor to int
            confidence = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = model.names[cls]  # Get class label from class index

            # Ensure the detected object is a license plate (if you have multiple classes)
            if label == 'License_Plate':
                # Crop the license plate area
                license_plate_crop = image[y1:y2, x1:x2]

                # Read the license plate number
                license_text, score = read_license_plate(license_plate_crop)

                # Draw bounding box on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display the detected license plate number
                if license_text:
                    cv2.putText(image, f'{license_text}', (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Optionally display the confidence score
                cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            # # Crop the license plate area
            # license_plate_crop = image[y1:y2, x1:x2]

            # # Read the license plate number
            # license_text, score = read_license_plate(license_plate_crop)

            # # Draw bounding box on the image
            # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # # Display the detected license plate number
            # if license_text:
            #     cv2.putText(image, f'{license_text}', (x1, y2 + 30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # # option to display the confidence score
            # cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image with predictions
    cv2.imwrite(save_path, image)
    print(f"Predicted image saved to: {save_path}")

if __name__ == '__main__':
    # Path to the image you want to test
    image = "license-plate-germany"
    decoder = "jpg"
    image_path = image+"."+decoder  # Replace with your image path
    # Path to your trained model weights
    model_path = 'license_plate_detector_GPU_NOv6.pt'  # Replace with your best model path
    # Path to save the predicted image
    save_path = image + "pred." + decoder  # Replace with your desired save path

    load_and_predict(image_path, model_path, save_path)
