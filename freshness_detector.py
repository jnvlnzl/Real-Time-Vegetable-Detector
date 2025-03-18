import cv2
import numpy as np
import pandas as pd

# Input from louie
vegetable = "potato"

# Load nutritional values
nutritional_values = pd.read_csv('nutritional_values.csv')

# Convert to HSV
image = cv2.imread("freshpotato.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

fresh_ranges = {
        "potato": ([20, 50, 50], [30, 255, 255]),  # yellow
        "carrot": ([5, 100, 100], [20, 255, 255]),  # orange
        "cabbage": ([35, 50, 50], [85, 255, 255])  # green
}

freshness_status = "rotten"

if vegetable in fresh_ranges:
    lower, upper = fresh_ranges[vegetable]
    lower_bound = np.array(lower, dtype=np.uint8)
    upper_bound = np.array(upper, dtype=np.uint8)
        
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)
        
    non_zero_pixels = cv2.countNonZero(mask)
    if non_zero_pixels > 10000:  # mema
        freshness_status = "fresh"

if freshness_status == "fresh":
    print(f"{vegetable.capitalize()} is fresh!")
    vegetable_data = nutritional_values[nutritional_values['name'].str.lower() == vegetable.lower()]
        
    if not vegetable_data.empty:
        print("Nutritional Information:")
        for nutrient, value in vegetable_data.iloc[0].items():
            print(f"{nutrient}: {value}")
    else:
        print("Nutritional data not found for this vegetable.")
else:
    print("Rotten vegetable detected.")

cv2.imshow("Original Image", image)
cv2.imshow("Mask", mask)
cv2.waitKey(0)