import cv2
import os
import replicate
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Set up authentication for Replicate
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
if not REPLICATE_API_TOKEN:
    raise ValueError("Please set the REPLICATE_API_TOKEN environment variable.")

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

def extract_frames(video_path, output_folder, frame_interval=2):
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    def save_frame(frame_count, image):
        cv2.imwrite(f"{output_folder}/frame{frame_count}.jpg", image)

    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            success, image = vidcap.read()
            if not success:
                break
            if frame_count % int(fps * frame_interval) == 0:
                executor.submit(save_frame, frame_count, image)
            frame_count += 1
    vidcap.release()

def process_frame(frame_path):
    with open(frame_path, "rb") as image_file:
        input = {"image": image_file, "mask_limit": 2}
        output = replicate.run(
            "lucataco/segment-anything-2:be7cbde9fdf0eecdc8b20ffec9dd0d1cfeace0832d4d0b58a071d993182e1be0",
            input=input
        )
    return output[0]

def download_image(url, save_path):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(save_path)

def highlight_firefighters(frames_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for frame in os.listdir(frames_folder):
        if frame.endswith(".jpg"):
            frame_path = os.path.join(frames_folder, frame)
            mask_url = process_frame(frame_path)
            output_path = os.path.join(output_folder, frame)
            download_image(mask_url, output_path)

def compile_video(input_folder, output_video_path, frame_rate=1):
    images = [img for img in os.listdir(input_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x[5:-4]))

    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def detect_firefighters(frame):
    net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CONFIG_PATH)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
            if confidence > 0.5:
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

def track_firefighter(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        cap.release()
        return

    bbox = detect_firefighters(frame)
    if not bbox:
        print("No firefighter detected")
        cap.release()
        return

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, tuple(bbox))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Check and download YOLO files
check_and_download_yolo_files()

# Paths
video_path = 'stock.mp4'
frames_folder = 'frames'
highlighted_frames_folder = 'highlighted_frames'
output_video_path = 'output_video.mp4'

# Extract frames, track firefighter, highlight frames, and compile video
extract_frames(video_path, frames_folder, frame_interval=2)
track_firefighter(video_path)
highlight_firefighters(frames_folder, highlighted_frames_folder)
compile_video(highlighted_frames_folder, output_video_path, frame_rate=1)
