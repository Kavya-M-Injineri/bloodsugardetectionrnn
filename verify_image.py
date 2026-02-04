import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained MobileNetV2
model = MobileNetV2(weights='imagenet')

def is_blood_sugar_related(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=5)[0]

    # Valid keywords for blood sugar related images in ImageNet
    valid_keywords = [
        'odometer', 'digital_clock', 'analog_clock', 'stopwatch', 'measuring_cup', 
        'scale', 'hand-held_computer', 'cellular_telephone', 'calculator',
        'oscilloscope', 'modem', 'monitor', 'screen', 'beaker', 'syringe', 
        'band_aid', 'magnetic_compass', 'odometer', 'space_heater', 'remote_control'
    ]
    
    # Keywords to explicitly reject
    invalid_keywords = [
        'plant', 'tree', 'flower', 'dog', 'cat', 'bird', 'animal', 'human', 
        'person', 'food', 'fruit', 'vegetable', 'dish', 'plate'
    ]

    top_label = decoded[0][1].lower()
    
    # Check if top label is in invalid list
    for inv in invalid_keywords:
        if inv in top_label:
            return False, f"Invalid input: {top_label}"

    # Check if any of top 5 match valid keywords
    is_valid = False
    for _, label, score in decoded:
        label = label.lower()
        if any(key in label for key in valid_keywords):
            is_valid = True
            break
    
    if not is_valid:
        return False, f"Invalid input: {top_label}"
    
    return True, "Valid input"

if __name__ == "__main__":
    # Test with local image
    import sys
    if len(sys.argv) > 1:
        valid, msg = is_blood_sugar_related(sys.argv[1])
        print(f"File: {sys.argv[1]} | Result: {valid} | Message: {msg}")
