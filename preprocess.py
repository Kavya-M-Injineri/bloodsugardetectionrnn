import cv2
import imutils
from imutils.perspective import four_point_transform
import numpy as np

def extract_digits(image_path, img_size=(28, 28), seq_len=5):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 1. Edge detection to find display
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            displayCnt = approx
            break
            
    if displayCnt is None:
        # Fallback to whole image if display not found
        warped = gray
    else:
        warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    
    # 2. Thresholding for digits
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # 3. Find digit contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # Filter based on size
        if w >= 5 and h >= 15:
            digitCnts.append(c)
            
    # Sort from left to right
    if len(digitCnts) > 0:
        digitCnts = sorted(digitCnts, key=lambda x: cv2.boundingRect(x)[0])
    
    # 4. Extract and resize each digit
    digit_images = []
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        roi = cv2.resize(roi, img_size)
        digit_images.append(roi.reshape(img_size[0], img_size[1], 1))
        
    # Pad or truncate to seq_len
    if len(digit_images) < seq_len:
        while len(digit_images) < seq_len:
            digit_images.append(np.zeros((img_size[0], img_size[1], 1), dtype=np.uint8))
    else:
        digit_images = digit_images[:seq_len]
        
    return np.array(digit_images) / 255.0

if __name__ == "__main__":
    # Test with p1.jpg
    seq = extract_digits("p1.jpg")
    if seq is not None:
        print(f"Extracted sequence shape: {seq.shape}")
    else:
        print("Failed to extract digits.")
