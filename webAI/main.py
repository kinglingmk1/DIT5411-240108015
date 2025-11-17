from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
import json
import os
import pickle

app = Flask(__name__)

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_H5_PATH = os.path.join(BASE_DIR, 'local_character_classes.h5')
MODEL_PKL_PATH = os.path.join(BASE_DIR, 'local_character_classes.pkl')
CLASS_LABELS_PATH = os.path.join(BASE_DIR, 'class_labels.json')

# Check if files exist
if not os.path.exists(MODEL_H5_PATH):
    raise FileNotFoundError(f"Model H5 file not found at: {MODEL_H5_PATH}")
if not os.path.exists(CLASS_LABELS_PATH):
    raise FileNotFoundError(f"Class labels file not found at: {CLASS_LABELS_PATH}")

print(f"Loading model from: {MODEL_H5_PATH}")
print(f"Model file exists: {os.path.exists(MODEL_H5_PATH)}")
print(f"Model file size: {os.path.getsize(MODEL_H5_PATH)} bytes")

# Load the model from H5 file
try:
    model = tf.keras.models.load_model(MODEL_H5_PATH)
    print("Model loaded successfully from .h5 file")
except Exception as e:
    print(f"Error loading .h5 file: {e}")
    # Try loading from pickle file as fallback
    if os.path.exists(MODEL_PKL_PATH):
        print(f"Attempting to load from pickle file: {MODEL_PKL_PATH}")
        with open(MODEL_PKL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully from .pkl file")
    else:
        raise

# Load class labels
with open(CLASS_LABELS_PATH, 'r', encoding='utf-8') as f:
    class_labels_data = json.load(f)
    class_labels = class_labels_data['labels']
    label_to_index = class_labels_data['label_to_index']
    index_to_label = class_labels_data['index_to_label']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.json
        image_data = data['image']
        
        # Remove the data:image/png;base64, prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGBA to handle transparency
        if image.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model's expected input size (50x50) with better quality
        image = image.resize((50, 50), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        top_predictions = [
            {
                'character': index_to_label[str(idx)],
                'confidence': float(predictions[0][idx] * 100)
            }
            for idx in top_indices
        ]
        
        return jsonify({
            'success': True,
            'predictions': top_predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("Starting Chinese Character Recognition Server...")
    print(f"Model loaded: {len(class_labels)} classes")
    print(f"Classes: {', '.join(class_labels)}")
    app.run(debug=True, host='0.0.0.0', port=5000)
