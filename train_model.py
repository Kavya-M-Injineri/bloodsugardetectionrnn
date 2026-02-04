import numpy as np
import cv2
import os
from rnn_model import build_rnn_model, IMG_SIZE, SEQ_LEN, NUM_CLASSES
from tensorflow.keras.utils import to_categorical

def generate_synthetic_digit(digit):
    # Very simple synthetic 7-segment digit generation using OpenCV
    img = np.zeros(IMG_SIZE, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(digit), (2, 22), font, 0.8, 255, 2)
    return img

def generate_dataset(num_samples=1000):
    X = []
    y = []
    for _ in range(num_samples):
        sequence_imgs = []
        sequence_labels = []
        # Generate a sequence of 5 random digits (0-9)
        # 10 is reserved for decimal or padding if needed, but here just digits
        for _ in range(SEQ_LEN):
            digit = np.random.randint(0, 10)
            img = generate_synthetic_digit(digit)
            # Add some noise
            noise = np.random.randint(0, 50, IMG_SIZE, dtype=np.uint8)
            img = cv2.add(img, noise)
            
            sequence_imgs.append(img.reshape(IMG_SIZE[0], IMG_SIZE[1], 1))
            sequence_labels.append(digit)
        
        X.append(sequence_imgs)
        y.append(to_categorical(sequence_labels, num_classes=NUM_CLASSES))
    
    return np.array(X) / 255.0, np.array(y)

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    X_train, y_train = generate_dataset(2000)
    X_val, y_val = generate_dataset(200)

    print("Building model...")
    model = build_rnn_model()

    print("Starting training...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

    model.save_weights('model_weights.weights.h5')
    print("Model weights saved to model_weights.weights.h5")
