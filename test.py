import cv2
import numpy as np
import joblib
from PIL import Image

# بارگذاری مدل آموزش‌دیده با joblib
model = joblib.load("adaboost_face_model.pkl")

# تابع تغییر اندازه تصویر
def resize_image(img, target_shape=(50, 37)):
    return np.array(Image.fromarray(img).resize(target_shape))

# تابع برای پیش‌بینی چهره در یک ناحیه
def predict_face_region(model, region):
    resized = resize_image(region)
    return model.predict([resized.flatten()])[0] == 1

# استریم ویدیو
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تبدیل به grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # اسکن تصویر با sliding window
    window_size = (50, 37)
    step_size = 10
    for y in range(0, gray.shape[0] - window_size[1], step_size):
        for x in range(0, gray.shape[1] - window_size[0], step_size):
            region = gray[y:y + window_size[1], x:x + window_size[0]]
            if region.shape[:2] == window_size and predict_face_region(model, region):
                cv2.rectangle(frame, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)

    # نمایش فریم
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
