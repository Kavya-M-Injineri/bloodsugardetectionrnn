import os
import cv2
import numpy as np
from rnn_model import build_rnn_model, SEQ_LEN
from preprocess import extract_digits

def evaluate():
    # Load model and weights
    print("Loading model...")
    model = build_rnn_model()
    weights_path = 'model_weights.weights.h5'
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Weights loaded from {weights_path}")
    else:
        print("Error: Weights file not found!")
        return

    samples = ['p1.jpg', 'p2.jpg', 'p5.jpg']
    results = {}

    for sample in samples:
        if not os.path.exists(sample):
            print(f"Skipping {sample}, file not found.")
            continue
            
        print(f"\nEvaluating {sample}...")
        sequence = extract_digits(sample, seq_len=SEQ_LEN)
        
        if sequence is None:
            print(f"Failed to extract digits from {sample}")
            continue
            
        # Predict
        batch_seq = np.expand_dims(sequence, axis=0) # Add batch dimension
        predictions = model.predict(batch_seq)
        
        # Parse results
        output_digits = []
        confidences = []
        
        for pred in predictions[0]:
            digit = np.argmax(pred)
            confidence = np.max(pred)
            output_digits.append(str(digit))
            confidences.append(confidence)
            
        prediction_str = "".join(output_digits)
        avg_confidence = np.mean(confidences)
        
        results[sample] = {
            'prediction': prediction_str,
            'avg_confidence': avg_confidence,
            'digit_confidences': confidences
        }
        
        print(f"  Prediction: {prediction_str}")
        print(f"  Average Confidence: {avg_confidence:.4f}")

    return results

if __name__ == "__main__":
    evaluate()
