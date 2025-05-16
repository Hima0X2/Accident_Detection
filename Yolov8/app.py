from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the YOLO model
model = YOLO("best.pt")

# Function to detect accidents in video
def detect_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Visualize results
        annotated_frame = results[0].plot()

        # Save or display using OpenCV
        cv2.imshow('YOLOv8 Detection', annotated_frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to detect accidents in image
def detect_from_image(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    annotated_image = results[0].plot()

    # Save or display the annotated image
    cv2.imshow('YOLOv8 Detection', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save file to the uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Perform accident detection on the uploaded file
        if file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            detect_from_video(file_path)
        elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            detect_from_image(file_path)
        else:
            return "Unsupported file format. Please upload a video or image."

        return render_template('index.html')

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)