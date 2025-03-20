from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import cv2
import os
import numpy as np
import joblib
from skimage.feature import hog
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

# Define paths and class labels
train_dir = 'Dataset/train'
class_labels = ['Bus', 'Car', 'Motorcycle', 'Truck']

# Define data augmentation pipeline
augmentation_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flip
    iaa.Affine(rotate=(-20, 20)),  # Rotate
    iaa.Affine(scale=(0.8, 1.2)),  # Scale
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Add Gaussian noise
    iaa.Multiply((0.8, 1.2))  # Change brightness
])

# Function to extract HOG features
def extract_hog_features(img):
    img = cv2.resize(img, (150, 150))  # Resize image
    features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Read images, apply data augmentation, and extract HOG features
data = []
true_labels = []

for label in class_labels:
    class_path = os.path.join(train_dir, label)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            continue
        # Original image
        features = extract_hog_features(img)
        data.append(features)
        true_labels.append(class_labels.index(label))
        # Augmented images
        for _ in range(5):  # Generate 5 augmented images per original image
            augmented_img = augmentation_pipeline(image=img)
            features = extract_hog_features(augmented_img)
            data.append(features)
            true_labels.append(class_labels.index(label))

# Convert to NumPy array
data = np.array(data)
true_labels = np.array(true_labels)

# Normalize the data
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)  # Adjust the number of components as needed
data_pca = pca.fit_transform(data)

# Define a range of k values to test
k_values = range(2, 11)  # Testing k values from 2 to 10
ari_scores = []

# Perform K-Means clustering for each k value and evaluate performance
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=48)
    predicted_labels = kmeans.fit_predict(data_pca)
    adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
    ari_scores.append(adjusted_rand)
    print(f"k: {k}, Adjusted Rand Index: {adjusted_rand:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, ari_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Adjusted Rand Index')
plt.title('Effect of k on Clustering Performance')
plt.grid(True)
plt.savefig('k_vs_ari.png')
plt.show()

# Save the best model 
best_k = k_values[np.argmax(ari_scores)]
best_kmeans = KMeans(n_clusters=best_k, random_state=48)
best_kmeans.fit(data_pca)
joblib.dump(best_kmeans, 'vehicle_classification_kmeans_model.pkl')
print(f"Best model with k={best_k} saved as 'vehicle_classification_kmeans_model.pkl'")