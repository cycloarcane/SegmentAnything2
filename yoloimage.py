import cv2
import os
import numpy as np

YOLO_CONFIG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
YOLO_CONFIG_PATH = "yolov3.cfg"
YOLO_WEIGHTS_PATH = "yolov3.weights"

def download_file(url, path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def check_and_download_yolo_files():
    if not os.path.exists(YOLO_CONFIG_PATH):
        print("Downloading YOLO configuration file...")
        download_file(YOLO_CONFIG_URL, YOLO_CONFIG_PATH)
    if not os.path.exists(YOLO_WEIGHTS_PATH):
        print("Downloading YOLO weights file...")
        download_file(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_PATH)

def detect_firefighter(image_path):
    net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CONFIG_PATH)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        return boxes[indexes[0]]
    else:
        return None

def create_mask(image_path, bbox):
    image = cv2.imread(image_path)
    mask = np.zeros_like(image)
    
    x, y, w, h = bbox
    mask[y:y+h, x:x+w] = 255
    
    mask_path = "mask.png"
    cv2.imwrite(mask_path, mask)
    
    return mask_path

def overlay_mask(image_path, mask_path, output_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the mask is binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Create a colored mask (red) to overlay
    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask == 255] = [0, 0, 255]
    
    # Overlay the mask on the original image
    overlayed_image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
    
    # Save the overlayed image
    cv2.imwrite(output_path, overlayed_image)

# Check and download YOLO files
check_and_download_yolo_files()

# Paths
image_path = 'frames/frame357.jpg'
overlayed_image_path = 'overlayed_image.jpg'

# Detect firefighter and create a mask
bbox = detect_firefighter(image_path)
if bbox:
    mask_path = create_mask(image_path, bbox)
    overlay_mask(image_path, mask_path, overlayed_image_path)
    print(f"Overlayed image saved at {overlayed_image_path}")
else:
    print("No firefighter detected in the image.")
