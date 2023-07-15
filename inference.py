import argparse
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Define the mapping from class index to ASCII code for 47 cases
CLASSES_TO_ASCII = {
    0: 48,   # '0'
    1: 49,   # '1'
    2: 50,   # '2'
    3: 51,   # '3'
    4: 52,   # '4'
    5: 53,   # '5'
    6: 54,   # '6'
    7: 55,   # '7'
    8: 56,   # '8'
    9: 57,   # '9'
    10: 65,  # 'A'
    11: 66,  # 'B'
    12: 67,  # 'C'
    13: 68,  # 'D'
    14: 69,  # 'E'
    15: 70,  # 'F'
    16: 71,  # 'G'
    17: 72,  # 'H'
    18: 74,  # 'J'
    19: 75,  # 'K'
    20: 76,  # 'L'
    21: 77,  # 'M'
    22: 78,  # 'N'
    23: 80,  # 'P'
    24: 81,  # 'Q'
    25: 82,  # 'R'
    26: 83,  # 'S'
    27: 84,  # 'T'
    28: 85,  # 'U'
    29: 86,  # 'V'
    30: 87,  # 'W'
    31: 88,  # 'X'
    32: 89,  # 'Y'
    33: 90,  # 'Z'
    34: 97,  # 'a'
    35: 98,  # 'b'
    36: 99,  # 'c'
    37: 100,  # 'd'
    38: 101,  # 'e'
    39: 102,  # 'f'
    40: 103,  # 'g'
    41: 104,  # 'h'
    42: 106,  # 'j'
    43: 107,  # 'k'
    44: 108,  # 'l'
    45: 109,  # 'm'
    46: 110,  # 'n'
}

# Define command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="Path to directory with image samples")
args = vars(ap.parse_args())

# Load the model
model = load_model("model.h5")

# Process each image in the input directory
for filename in os.listdir(args["input"]):
    # Check if the file is an image
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Load the image and preprocess it
    image = cv2.imread(os.path.join(args["input"], filename), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = image.astype("float32") / 255.0

    # Make predictions and convert them to ASCII codes
    predictions = model.predict(image)
    predictions = np.argmax(predictions, axis=1)
    text = "".join([chr(CLASSES_TO_ASCII[prediction]) for prediction in predictions])
    print(f"{text}, {filename}: ")
    #"".join([chr(CLASSES_TO_ASCII[prediction]) for prediction in predictions])
    #([CLASSES_TO_ASCII[prediction] for prediction in predictions][0])
