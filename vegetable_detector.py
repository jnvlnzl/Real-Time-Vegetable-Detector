import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image, mask

def detect_vegetable(image, mask):
    isolated = cv2.bitwise_and(image, image, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return "No vegetable detected", isolated

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    if 0.78 < circularity:  
        vegetable = "Cabbage"
    elif 0.75 < circularity: 
        vegetable = "Potato"
    elif 0.7 < circularity:  
        vegetable = "Carrot"
    else:
        vegetable = "Unknown Vegetable"
    
    return vegetable, isolated

image_path = "freshcarrot.jpg"
image, mask = preprocess_image(image_path)

vegetable, isolated = detect_vegetable(image, mask)

cv2.imshow("Original Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Isolated Vegetable", isolated)
print(vegetable)
cv2.waitKey(0)
cv2.destroyAllWindows()
