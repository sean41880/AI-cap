from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

test_dir = 'Dataset/test'
class_labels = ['Bus', 'Car', 'Motorcycle', 'Truck']

svm_model = joblib.load('vehicle_classification_svm_model.pkl')
print("Model loaded successfully!")

# Function to extract HOG features
def extract_hog_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.resize(img, (150, 150))  # Resize image
    features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Load data from a directory
def load_data(directory):
    data = []
    labels = []
    for label in class_labels:
        class_path = os.path.join(directory, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            features = extract_hog_features(img_path)
            data.append(features)
            labels.append(class_labels.index(label))
    return np.array(data), np.array(labels)

# Load test data
X_test, y_test = load_data(test_dir)

# Evaluate the model on test data
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('svm_confusion_matrix.png')
plt.show()

# Classification report
class_report = classification_report(y_test, y_test_pred, target_names=class_labels)
print("Classification Report:")
print(class_report)