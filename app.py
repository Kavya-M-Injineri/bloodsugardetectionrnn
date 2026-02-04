import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from rnn_model import build_rnn_model, SEQ_LEN
from preprocess import extract_digits
from verify_image import is_blood_sugar_related

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global model variable
model = None

def get_model():
    global model
    if model is None:
        model = build_rnn_model()
        if os.path.exists('model_weights.weights.h5'):
            model.load_weights('model_weights.weights.h5')
            print("Loaded model weights from model_weights.weights.h5")
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # 1. Image Type Verification
    is_valid, message = is_blood_sugar_related(filepath)
    if not is_valid:
        # Output exactly the failure message (placeholder for missing user spec)
        return jsonify({'error': 'Invalid image type detected. Please upload a blood sugar related image.'}), 400

    # 2. Preprocess image using robust logic
    sequence = extract_digits(filepath, seq_len=SEQ_LEN)
    if sequence is None:
        return jsonify({'error': 'Failed to process digits from image.'}), 400
    
    # 3. Predict
    sequence = np.expand_dims(sequence, axis=0) # Add batch dimension
    model = get_model()
    predictions = model.predict(sequence)
    
    # Convert predictions to digits (argmax)
    output_digits = []
    for pred in predictions[0]:
        digit = np.argmax(pred)
        output_digits.append(str(digit))
    
    return jsonify({'blood_sugar': "".join(output_digits)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
