import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
from PIL import Image

def resize_image(img, target_shape=(50, 37)):
    return np.array(Image.fromarray(img).resize(target_shape))

def load_data(data_dir, label):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        img = np.array(Image.open(filepath).convert('L'))
        img_resized = resize_image(img)
        images.append(img_resized.flatten())
        labels.append(label)
    return images, labels

faces_dir = "C://Users/Amir_moradi/Desktop/face"
non_faces_dir = "C://Users/Amir_moradi/Desktop/not_face"

face_images, face_labels = load_data(faces_dir, 1)
non_face_images, non_face_labels = load_data(non_faces_dir, 0)

X = np.array(face_images + non_face_images)
y = np.array(face_labels + non_face_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_estimator = DecisionTreeClassifier(max_depth=2)
model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50)
model.fit(X_train, y_train)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'estimator__max_depth': [2, 3, 5]
}

grid_search = GridSearchCV(
    AdaBoostClassifier(estimator=base_estimator),
    param_grid,
    scoring='accuracy',
    cv=3
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

joblib.dump(model, "adaboost_face_model.pkl")
print("Model trained and saved successfully!")

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")
