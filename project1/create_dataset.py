import kagglehub
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import shutil

'''
# Download the dataset
kagglehub.dataset_download("kaggleashwin/vehicle-type-recognition")
'''

dataset_path = 'Dataset'
categories = ['Bus', 'Car', 'Motorcycle', 'Truck']

# Create train and test directories
for category in categories:
    os.makedirs(os.path.join(dataset_path, 'train', category), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'test', category), exist_ok=True)

# Split the dataset
for category in categories:
    category_path = os.path.join(dataset_path, category)
    images = os.listdir(category_path)
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=3)

    for image in train_images:
        shutil.copy(os.path.join(category_path, image), os.path.join(dataset_path, 'train', category, image))

    for image in test_images:
        shutil.copy(os.path.join(category_path, image), os.path.join(dataset_path, 'test', category, image))

print("Dataset split into training and testing sets completed.")