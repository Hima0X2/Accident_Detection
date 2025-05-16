from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import shutil

# Initialize Flask app
app = Flask(__name__)

# Define upload folder and ensure it's inside static for browser access
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the focal loss function (needed to load the model)
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Apply sigmoid if needed
        if y_pred.shape[-1] == 1:
            y_pred = tf.sigmoid(y_pred)
            
        # Class 1 and Class 0 probabilities
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        
        # Calculate focal loss
        loss = - alpha_factor * modulating_factor * tf.math.log(tf.clip_by_value(p_t, 1e-7, 1.0))
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# Load the model with custom objects
try:
    model = load_model('models/final_balanced_model.h5', 
                    custom_objects={'focal_loss_fixed': focal_loss()})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set to None to handle errors gracefully

# Load optimal threshold if available
threshold = 0.5  # Default threshold
try:
    with open('models/optimal_threshold.txt', 'r') as f:
        threshold = float(f.read().strip())
        print(f"Loaded optimal threshold: {threshold}")
except:
    print(f"Using default threshold: {threshold}")

# Function to preprocess images for the CNN model
def preprocess_image(image):
    # Resize image to match input size of the model (224x224)
    image = cv2.resize(image, (224, 224))
    # Ensure the image has 3 channels (RGB)
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize image
    return image

# Function to detect accident using the CNN model
def detect_accident(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None
    
    # Save a copy of the original image for display
    original_filename = 'original_' + os.path.basename(image_path)
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    cv2.imwrite(original_path, image)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict using the model
    if model is not None:
        predictions = model.predict(preprocessed_image)
        print("Raw Model Predictions:", predictions)
        
        # Use the loaded threshold for classification
        if predictions[0][0] < threshold:
            result = f"Accident Detected ({predictions[0][0]:.2f})"
            color = (0, 0, 255)  # Red color in BGR format
            has_accident = True
        else:
            result = f"No Accident Detected ({predictions[0][0]:.2f})"
            color = (0, 255, 0)  # Green color in BGR format
            has_accident = False
    else:
        result = "Model not loaded, cannot predict"
        color = (255, 0, 0)  # Blue color
        has_accident = None

    # Annotate the image with the result
    cv2.putText(image, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Save the annotated image
    result_filename = 'result_' + os.path.basename(image_path)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, image)
    
    return result_filename, has_accident, original_filename

# Function to detect accident in video file
def detect_accident_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Save the original video for display
    original_filename = 'original_' + os.path.basename(video_path)
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    shutil.copy(video_path, original_path)
    
    # Define output video file
    result_filename = 'result_' + os.path.basename(video_path)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    
    # Use MP4V codec for maximum compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
    
    frame_count = 0
    accident_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Process every 5th frame to speed up processing
        process_this_frame = (frame_count % 5 == 0)
        
        if process_this_frame or frame_count == 1:
            # Preprocess the frame
            preprocessed_frame = preprocess_image(frame)
            
            # Predict using the model
            if model is not None:
                predictions = model.predict(preprocessed_frame, verbose=0)
                
                # Use the loaded threshold for classification
                if predictions[0][0] <= threshold:
                    result = f"Accident Detected ({predictions[0][0]:.2f})"
                    color = (0, 0, 255)  # Red color
                    accident_frames += 1
                else:
                    result = f"No Accident Detected ({predictions[0][0]:.2f})"
                    color = (0, 255, 0)  # Green color
            else:
                result = "Model not loaded, cannot predict"
                color = (255, 0, 0)  # Blue color
            
            # Annotate the frame with the result
            cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Write the frame to output video
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Generate thumbnail from the video for display
    thumbnail_filename = 'thumb_' + os.path.basename(video_path).split('.')[0] + '.jpg'
    thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], thumbnail_filename)
    
    # Extract a frame from the middle of the video for thumbnail
    cap = cv2.VideoCapture(result_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count // 2, 50))  # Middle frame or frame 50
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(thumbnail_path, frame)
    cap.release()
    
    has_accident = (accident_frames > 0)
    
    return result_filename, has_accident, original_filename, thumbnail_filename

# Route to render the index page (upload form)
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Create upload folder if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Process based on file type
        if file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            result_filename, has_accident, original_filename, thumbnail_filename = detect_accident_video(file_path)
            if result_filename is None:
                return "Error processing video file."
            
            return render_template('result_video.html',
                                   original_filename=original_filename,
                                   result_filename=result_filename,
                                   thumbnail_filename=thumbnail_filename,
                                   has_accident=has_accident)
        
        # elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        #     result_filename, has_accident, original_filename = detect_accident(file_path)
        #     if result_filename is None:
        #         return "Error processing image file."
            
        #     return render_template('result_image.html',
        #                            original_filename=original_filename,
        #                            result_filename=result_filename,
        #                            has_accident=has_accident)
        else:
            return "Unsupported file format. Please upload a video"
    
    return redirect(request.url)

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)