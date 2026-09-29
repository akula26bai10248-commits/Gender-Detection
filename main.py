import cv2
import numpy as np
import urllib.request
import os
import sys
from collections import deque
import statistics

# ============================================================
# GENDER DETECTION - SINGLE FILE REAL-TIME WEBCAM PROJECT
# ============================================================

# -----------------------------
# MODEL FILES & URLS
# -----------------------------
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
GENDER_PROTO_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/Gender%20Detection/gender_deploy.prototxt"
GENDER_MODEL_URL = "https://github.com/spmallick/learnopencv/raw/master/Gender%20Detection/gender_net.caffemodel"

def download_file(url, filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return
    print(f"[DOWNLOAD] {filename}")
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        print(f"[ERROR] Could not download {filename}\n{e}")
        sys.exit(1)

print("\n==========================================")
print("     GENDER DETECTION AI")
print("==========================================\n")
print("Checking models...")

download_file(FACE_PROTO_URL, FACE_PROTO)
download_file(FACE_MODEL_URL, FACE_MODEL)
download_file(GENDER_PROTO_URL, GENDER_PROTO)
download_file(GENDER_MODEL_URL, GENDER_MODEL)

try:
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
except Exception as e:
    print(f"\n[ERROR] Could not load AI models.\n{e}")
    sys.exit(1)

GENDER_LIST = ["Male", "Female"]

print("Starting webcam...")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("\n[ERROR] Webcam could not be opened.")
    sys.exit(1)

# -----------------------------
# NEW: TEMPORAL BUFFER
# Stores the last 15 predictions to stop the text from flickering
# -----------------------------
gender_buffer = deque(maxlen=15)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.60:
            continue

        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        startX, startY, endX, endY = box.astype(int)

        # -----------------------------
        # NEW: BOUNDING BOX PADDING
        # Expand the crop area by 20% to capture hair, jaw, and neck
        # -----------------------------
        pad_x = int((endX - startX) * 0.20)
        pad_y = int((endY - startY) * 0.20)

        p_startX = max(0, startX - pad_x)
        p_startY = max(0, startY - pad_y)
        p_endX = min(width - 1, endX + pad_x)
        p_endY = min(height - 1, endY + pad_y)

        face = frame[p_startY:p_endY, p_startX:p_endX]
        if face.size == 0:
            continue

        # -----------------------------
        # NEW: ANTI-MOIRÉ FILTER
        # Blur the face slightly to remove phone screen pixels so 
        # the AI doesn't mistake screen grids for a male beard
        # -----------------------------
        face_smoothed = cv2.GaussianBlur(face, (5, 5), 0)

        # Pass the padded, smoothed face to the model
        gender_blob = cv2.dnn.blobFromImage(face_smoothed, 1.0, (227, 227), 
                                            (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_net.setInput(gender_blob)
        gender_predictions = gender_net.forward()

        gender_index = gender_predictions[0].argmax()
        current_gender = GENDER_LIST[gender_index]
        
        # Add the current frame's guess to our rolling memory
        gender_buffer.append(current_gender)

        # -----------------------------
        # NEW: STABLE OUTPUT
        # Only display the most common prediction over the last 15 frames
        # -----------------------------
        stable_gender = statistics.mode(gender_buffer)

        # Draw visual elements based on the padded box
        cv2.rectangle(frame, (p_startX, p_startY), (p_endX, p_endY), (0, 255, 0), 2)
        
        label = f"{stable_gender}"
        
        # Change text color dynamically (Green for Male, Magenta for Female)
        color = (0, 255, 0) if stable_gender == "Male" else (255, 0, 255)

        cv2.putText(frame, label, (p_startX, max(p_startY - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Real-Time Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
