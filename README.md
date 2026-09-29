# Real-Time Gender Detection Using OpenCV and Deep Learning

## 1. Project Overview

This project is a **real-time gender detection system** developed using Python and OpenCV. It uses a webcam to capture live video, detects faces using a pre-trained deep-learning face detector, and predicts the gender of each detected face using a pre-trained gender classification model.

The application displays the detected face with a bounding box and shows the predicted gender directly on the webcam window.

The project also includes a temporal prediction buffer to reduce flickering and improve the stability of the displayed prediction.

---

## 2. Features

- Real-time webcam-based face detection.
- Gender classification as **Male** or **Female**.
- Automatic downloading of required AI model files.
- Face detection using OpenCV DNN.
- Bounding-box padding to provide a larger face region to the gender model.
- Gaussian blur preprocessing.
- 15-frame prediction buffer for stable output.
- Real-time bounding boxes and gender labels.
- Simple single-file Python implementation.

---

## 3. Technologies Used

- **Python 3**
- **OpenCV**
- **NumPy**
- **OpenCV DNN**
- Pre-trained Caffe deep-learning models
- Computer webcam

---

## 4. Project Structure

```text
gender-detection/
│
├── gender_detection.py
├── deploy.prototxt
├── res10_300x300_ssd_iter_140000.caffemodel
├── gender_deploy.prototxt
├── gender_net.caffemodel
└── README.md
```

The four model files are downloaded automatically by the Python program if they are not already present.

---

# 5. Requirements

Before running the project, make sure the following are installed:

- Python 3.8 or newer
- Working webcam
- Internet connection for the initial model download
- pip package manager

The Python program imports OpenCV, NumPy, `urllib`, `os`, `sys`, `deque`, and `statistics`.

---

# 6. Environment Setup

## Step 1: Install Python

Download and install Python 3 from the official Python website.

During installation on Windows, make sure to enable:

```text
Add Python to PATH
```

Verify the installation:

```bash
python --version
```

or:

```bash
python3 --version
```

You should see a Python 3 version.

---

## Step 2: Create a Project Directory

Create a folder for the project:

```bash
mkdir gender-detection
cd gender-detection
```

Place the following file inside this folder:

```text
gender_detection.py
```

---

# 7. Create a Virtual Environment

It is recommended to use a virtual environment so that the project's dependencies remain isolated.

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

After activation, your terminal should indicate that the virtual environment is active.

---

# 8. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required Python packages:

```bash
pip install opencv-python numpy
```

The main external dependencies used by the project are OpenCV and NumPy.

---

# 9. Configuration

No separate configuration file is required.

The program defines the four required model files:

```text
deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel
gender_deploy.prototxt
gender_net.caffemodel
```

It also contains the download URLs for these models.

When the program starts, it checks whether each model file exists. If a file is missing, the program automatically downloads it.

Therefore, an internet connection is required the first time the program is executed.

---

# 10. Run the Project

Make sure the virtual environment is activated.

Run:

```bash
python gender_detection.py
```

The program will:

1. Display the project startup message.
2. Check for the required AI models.
3. Download missing models.
4. Load the face-detection model.
5. Load the gender-classification model.
6. Open the webcam.
7. Detect faces in real time.
8. Classify the detected faces.
9. Display the predicted gender.
10. Continue processing until the user exits.

The program loads both models using OpenCV's `readNetFromCaffe()` function.

---

# 11. Using the Application

After starting the program, a webcam window named:

```text
Real-Time Gender Detection
```

will appear.

Position your face in front of the camera.

When a face is detected, the program draws a bounding box around it and displays the predicted gender above the box.

---

# 12. Exit the Application

To stop the program, press:

```text
Q
```

The program checks for the `q` key during every webcam frame.

The webcam is then released and the OpenCV windows are closed.

---

# 13. How the System Works

The application first captures a frame from the webcam.

The frame is resized and converted into a DNN blob before being passed to the face-detection network.

The face detector examines the frame and returns detected face regions with confidence values. Only detections with a confidence of at least **0.60** are processed.

The detected face region is then expanded by approximately 20% to provide additional facial context.

The face is slightly blurred using a Gaussian filter and converted into a 227 × 227 input blob for the gender-classification network.

The gender model produces predictions, and the class with the highest prediction value is selected:

```text
Male
Female
```

These labels are defined in the program as:

```python
GENDER_LIST = ["Male", "Female"]
```



---

# 14. Prediction Stabilization

Real-time predictions can sometimes change rapidly between consecutive frames.

To reduce this problem, the project maintains a rolling buffer containing the most recent **15 predictions**.

```python
gender_buffer = deque(maxlen=15)
```

The program then selects the most common prediction in the buffer and displays that result.

This helps prevent the displayed label from rapidly flickering between predictions.

---

# 15. Troubleshooting

## Webcam Does Not Open

If you see:

```text
[ERROR] Webcam could not be opened.
```

check that:

- Your webcam is connected.
- No other application is using the webcam.
- Python has permission to access the camera.
- The correct camera is selected by the system.

The program uses:

```python
camera = cv2.VideoCapture(0)
```

to access the default webcam.

---

## Model Download Fails

If model downloading fails:

1. Check your internet connection.
2. Make sure the model URLs are accessible.
3. Run the program again.
4. Check whether your firewall or network is blocking the download.

The program terminates if a required model cannot be downloaded.

---

## OpenCV Installation Error

Try:

```bash
python -m pip install --upgrade pip
pip install --upgrade opencv-python numpy
```

Then run:

```bash
python gender_detection.py
```

---

## Camera Window Is Slow

Real-time performance depends on:

- CPU performance
- Webcam resolution
- Number of detected faces
- OpenCV version
- Available system resources

Reducing the webcam resolution can help improve performance if required.

---

# 16. Important Notes

This project demonstrates **AI-based gender classification from facial images**. The displayed classification is a model prediction and should not be interpreted as a definitive determination of a person's gender identity.

The project should be used responsibly and with appropriate consent when processing other people's images.

---

# 17. Quick Start

For users who already have Python installed:

```bash
git clone <your-repository-url>
cd gender-detection

python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install opencv-python numpy
```

Run:

```bash
python gender_detection.py
```

Allow the application to access the webcam.

Press **Q** to exit.

---

# 18. Summary

This project provides a simple real-time gender classification application using Python, OpenCV, webcam input, and pre-trained deep-learning models. The implementation automatically manages the required model files and uses face detection, preprocessing, gender classification, and temporal smoothing to produce a stable real-time display.
