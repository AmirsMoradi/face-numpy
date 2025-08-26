import cv2
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt

model = joblib.load("adaboost_face_model.pkl")

def resize_image(img, target_shape=(50, 37)):
    return np.array(Image.fromarray(img).resize(target_shape))

def predict_face_region(model, region):
    resized = resize_image(region)
    return model.predict([resized.flatten()])[0] == 1

cap = cv2.VideoCapture(0)

window_sizes = [(50, 37), (75, 55), (100, 75)]
step_size = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    gray = cv2.equalizeHist(gray)

    for window_size in window_sizes:
        for y in range(0, gray.shape[0] - window_size[1], step_size):
            for x in range(0, gray.shape[1] - window_size[0], step_size):
                region = gray[y:y + window_size[1], x:x + window_size[0]]
                if region.shape[:2] == window_size and predict_face_region(model, region):
                    frame[y:y + window_size[1], x:x + 3] = [255, 255, 0]
                    frame[y:y + window_size[1], x + window_size[0] - 3:x + window_size[0]] = [255, 255, 0]
                    frame[y:y + 3, x:x + window_size[0]] = [255, 255, 0]
                    frame[y + window_size[1] - 3:y + window_size[1], x:x + window_size[0]] = [255, 255, 0]

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
    plt.axis('off')
    plt.pause(0.01)  # Add a short delay for real-time display

    # Exit loop if a key is pressed
    if plt.waitforbuttonpress(0.01):
        break

cap.release()
plt.close()
