import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import zipfile
import shutil
import tensorflow as tf
import pickle
import random
import json
import keras
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
sys.stdin.reconfigure(encoding="utf-8") #I hate Windows Python
path = "D:\\TACI\\Traditional_Chinese_Data\\" #chagne to that path you unzipped the data, name should be same as Traditional_Chinese_Data
output_folder = "D:\\TACI\\output2\\" #create one or change to your output folder also line 69
wordCount = 0
sampleNeeded = 40
for entry in os.listdir(path):
    if os.path.isdir(os.path.join(path, entry)):
        print(f"Processing character: {entry}")
        wordCount = 0
        for file in os.listdir(os.path.join(path, entry)):
            if not os.path.exists(output_folder + entry):
                os.makedirs(output_folder + entry)
            #if exsist then break
            img = cv2.imdecode(np.fromfile(os.path.join(path, entry, file), dtype=np.uint8), cv2.IMREAD_COLOR)
            #check is img exsist in output folder
            if not os.path.exists(os.path.join(output_folder + entry, f"{entry}_{wordCount}_t1.jpg")):
                cv2.imencode('.jpg', img)[1].tofile(os.path.join(output_folder + entry, f"{entry}_{wordCount}_t1.jpg"))
            if not os.path.exists(os.path.join(output_folder + entry, f"{entry}_{wordCount}_t2.jpg")):
                height , width = img.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2), 90, 1)
                img_rotated =  cv2.warpAffine(img, rotation_matrix , (width,height))
                cv2.imencode('.jpg', img_rotated)[1].tofile(os.path.join(output_folder + entry, f"{entry}_{wordCount}_t2.jpg"))
            if not os.path.exists(os.path.join(output_folder + entry, f"{entry}_{wordCount}_t3.jpg")):
                rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2), 270, 1)
                img_rotated =  cv2.warpAffine(img, rotation_matrix , (width,height))
                cv2.imencode('.jpg', img_rotated)[1].tofile(os.path.join(output_folder + entry, f"{entry}_{wordCount}_t3.jpg"))
            if not os.path.exists(os.path.join(output_folder + entry, f"{entry}_{wordCount}_t4.jpg")):
                img_scaleBig = cv2.resize(img, (100,100), interpolation=cv2.INTER_LINEAR)
                cv2.imencode('.jpg', img_scaleBig)[1].tofile(os.path.join(output_folder + entry, f"{entry}_{wordCount}_t4.jpg"))
            if not os.path.exists(os.path.join(output_folder + entry, f"{entry}_{wordCount}_t5.jpg")):
                rows, cols = img.shape[:2]
                M = np.float32([[1, 0.5, 0], [0, 0.5, 1]])
                img_translated = cv2.warpAffine(img, M, (cols, rows))
                cv2.imencode('.jpg', img_translated)[1].tofile(os.path.join(output_folder + entry, f"{entry}_{wordCount}_t5.jpg"))
            wordCount += 1
            if wordCount >= sampleNeeded:
                break
#from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Drop
print(tf.__version__)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print("GPUs Available:")
    for gpu in gpu_devices:
        # Get details for each GPU, including its name
        gpu_details = tf.config.experimental.get_device_details(gpu)
        print(f"  Device Name: {gpu_details['device_name']}")
else:
    print("No GPUs detected by TensorFlow.")
size = 50
bigSize = 100
training_data = []
output_folder = "D:\\TACI\\output2\\" #remember change here
print("Loading training data...")
character_count = 0
for entry in os.listdir(output_folder):
    if os.path.isdir(os.path.join(output_folder, entry)):
        character_count += 1
        print(f"Loading character {character_count}: {entry}")
        for file in os.listdir(os.path.join(output_folder, entry)):
            img = cv2.imdecode(np.fromfile(os.path.join(output_folder, entry, file), dtype=np.uint8), cv2.IMREAD_COLOR)
            img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            training_data.append([img_resized, entry])
        if character_count >= 1500: #test limit # my enviroment only handle 1500 classes more will cause ram not enough
            break
print(f"\nTotal characters loaded: {character_count}")
print(f"Total training samples: {len(training_data)}")
random.shuffle(training_data)
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, size, size, 3)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_encoded = np.array(y_encoded)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
X = X / 255.0
print(f"\nX shape: {X.shape}")
print(f"y shape: {y_encoded.shape}")
print(f"Number of classes: {len(le.classes_)}")
model = models.Sequential([
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(size, size, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Third conv block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Fourth conv block
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Fully connected layers
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(le.classes_), activation='softmax')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

print("\nModel Summary:")
model.summary()

print("\nStarting training...")
history = model.fit(
    X, y_encoded, 
    epochs=100,
    batch_size=12,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
#save model
model.save('local_character_classes.h5')
with open('local_character_classes.pkl', 'wb') as f:
    pickle.dump(model, f)
model.save('local_chinese_character_recognition_model.keras')

print("\nSaving model configuration as JSON...")

# Create json configuration
model_config = {
    "model_info": {
        "name": "Traditional Chinese Character Recognition Model",
        "version": "1.0",
        "input_shape": [size, size, 3],
        "num_classes": len(le.classes_),
        "total_training_samples": len(training_data),
        "model_architecture": "CNN with 4 conv blocks + dense layers"
    },
    "training_config": {
        "epochs_trained": len(history.history['accuracy']),
        "batch_size": 32,
        "validation_split": 0.15,
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "final_training_accuracy": float(history.history['accuracy'][-1]),
        "final_validation_accuracy": float(history.history['val_accuracy'][-1]),
        "final_training_loss": float(history.history['loss'][-1]),
        "final_validation_loss": float(history.history['val_loss'][-1])
    },
    "class_labels": {
        "classes": le.classes_.tolist(),
        "num_classes": len(le.classes_),
        "encoding": "LabelEncoder (sklearn)"
    },
    "preprocessing": {
        "image_size": size,
        "normalization": "0-1 range (divide by 255)",
        "color_mode": "RGB"
    },
    "files": {
        "keras_model": "local_chinese_character_recognition_model.keras",
        "h5_model": "local_character_classes.h5",
        "pickle_model": "local_character_classes.pkl",
        "label_encoder": "label_encoder.pkl",
        "config": "model_config.json"
    }
}

# Save json
with open('model_config.json', 'w', encoding='utf-8') as f:
    json.dump(model_config, f, indent=4, ensure_ascii=False)
class_labels_data = {
    "labels": le.classes_.tolist(),
    "label_to_index": {label: int(idx) for idx, label in enumerate(le.classes_)},
    "index_to_label": {int(idx): label for idx, label in enumerate(le.classes_)}
}
with open('class_labels.json', 'w', encoding='utf-8') as f:
    json.dump(class_labels_data, f, indent=4, ensure_ascii=False)
print("\nTraining complete!")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
# Load model
# Load configuration
with open('model_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
# Load class labels
with open('class_labels.json', 'r', encoding='utf-8') as f:
    labels_data = json.load(f)
# Load trained model
loaded_model = load_model('local_chinese_character_recognition_model.keras')
print("Model loaded successfully!")
print(f"Number of classes: {config['model_info']['num_classes']}")
print(f"Input shape: {config['model_info']['input_shape']}")
print(f"Training accuracy: {config['training_config']['final_training_accuracy']:.4f}")
print(f"Validation accuracy: {config['training_config']['final_validation_accuracy']:.4f}")
print(f"\nTotal characters: {len(labels_data['labels'])}")
print(f"First 10 characters: {labels_data['labels'][:10]}")
def predict_character(image_path, model, label_encoder, image_size=64, top_k=3):
    """Predict character from image and show top K predictions"""
    # Read and preprocess image
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    predictions = model.predict(img_batch, verbose=0)[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    print(f"Top {top_k} predictions:")
    for idx in top_indices:
        character = label_encoder.classes_[idx]
        confidence = predictions[idx] * 100
        print(f"{character}: {confidence:.2f}%")
    return label_encoder.classes_[top_indices[0]]
total_chars = len([d for d in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, d))])
print(f"Total characters available in dataset: {total_chars}")
print(f"\nYou can train on up to {total_chars} characters!")