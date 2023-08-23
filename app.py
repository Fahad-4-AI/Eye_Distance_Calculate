import cv2
import dlib
import math
import base64
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/preview', methods=['POST'])
def preview_page():
    personal_image = request.files['personalPhoto']
    
    # Save the uploaded images temporarily
    personal_image_path = 'static/' + personal_image.filename
    personal_image.save(personal_image_path)
    
    return render_template('preview.html', personal_image=personal_image_path)

@app.route('/next_page', methods=['POST'])
def next_page():
    personal_image = request.form['personal_image']
    
    # Load the image using OpenCV
    image = cv2.imread(personal_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) corresponding to the detected face
        face_roi = gray[y:y+h, x:x+w]
        # Load the pre-trained facial landmarks detector
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Detect facial landmarks for the ROI
        landmarks = predictor(face_roi, dlib.rectangle(0, 0, w, h))

        # Extract the coordinates of pupils (landmarks 36 and 45)
        x1, y1 = landmarks.part(36).x, landmarks.part(36).y  # Left pupil
        x2, y2 = landmarks.part(45).x, landmarks.part(45).y  # Right pupil
        # Calculate the distance between pupils in pixels
        distance_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        distance_mm = distance_pixels / 4.1
        distance_mm = round(distance_mm, 2)
        
        # Draw rectangles around eyes
        cv2.rectangle(image, (x + x1 - 10, y + y1 - 10), (x + x1 + 10, y + y1 + 10), (0, 255, 0), 2)
        cv2.rectangle(image, (x + x2 - 10, y + y2 - 10), (x + x2 + 10, y + y2 + 10), (0, 255, 0), 2)

    # Convert the modified image to base64
    _, img_encoded = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return render_template('next_page.html', personal_image=image_base64, distance_mm=distance_mm)

if __name__ == '__main__':
    app.run(debug=True)
