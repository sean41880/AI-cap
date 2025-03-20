from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Define paths and class labels
train_dir = 'Dataset/train'
class_labels = ['Bus', 'Car', 'Motorcycle', 'Truck']

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

# Load training data
X, y = load_data(train_dir)

# Perform cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
best_auc = 0
best_model = None

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Train SVM model
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    # Predict probabilities
    y_val_pred_prob = svm_model.predict_proba(X_val)

    # Calculate AUC-ROC
    auc = roc_auc_score(y_val, y_val_pred_prob, multi_class='ovr')
    auc_scores.append(auc)

    # Save the model if it has the best AUC-ROC score
    if auc > best_auc:
        best_auc = auc
        best_model = svm_model
        joblib.dump(svm_model, f'best_svm_model_fold_{fold}.pkl')

# Print AUC-ROC scores
print(f"AUC-ROC scores: {auc_scores}")
print(f"Mean AUC-ROC: {np.mean(auc_scores):.2f}")

# Save the best model
if best_model:
    joblib.dump(best_model, 'best_vehicle_classification_svm_model.pkl')
    print("Best model saved as 'best_vehicle_classification_svm_model.pkl'")