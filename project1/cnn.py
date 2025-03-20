import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths
dataset_dir = 'Dataset/train'
dataset_dir_test = 'Dataset/test'
class_labels = ['Bus', 'Car', 'Motorcycle', 'Truck']

# Image data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen_without_augmentation = ImageDataGenerator(
    rescale=1./255
)

# Load data
def load_data(directory):
    data = []
    labels = []
    for label in class_labels:
        class_path = os.path.join(directory, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            data.append(img_array)
            labels.append(class_labels.index(label))
    return np.array(data), np.array(labels)

X_train, y_train = load_data(dataset_dir)
X_test, y_test = load_data(dataset_dir_test)

# Convert labels to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=4)

# Define the model with adjustable hyper-parameters
def create_model(learning_rate=0.001, dropout_rate=0.5, dense_units=512):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  # Freeze the pre-trained layers

    model = Sequential([
        base_model,
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

def evaluate_data_amounts(data_amounts):
    accuracies = []

    for amount in data_amounts:
        # Use a subset of the training data
        subset_size = int(len(X_train) * amount)
        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]
        y_train_cat_subset = tf.keras.utils.to_categorical(y_train_subset, num_classes=4)

        # Create data generators
        train_generator = datagen.flow(X_train_subset, y_train_cat_subset, batch_size=32)
        test_generator = datagen.flow(X_test, y_test_cat, batch_size=32, shuffle=False)

        # Create and train the model
        model = create_model()
        history = model.fit(train_generator, validation_data=test_generator, epochs=10)

        # Get the final test accuracy
        test_accuracy = history.history['val_accuracy'][-1]
        accuracies.append(test_accuracy)

        print(f"Training data amount: {amount*100}%, Test Accuracy: {test_accuracy:.2f}")

    # Plot the graph of data amount vs. accuracy
    plt.figure(figsize=(10, 6))
    plt.plot([amount*100 for amount in data_amounts], accuracies, marker='o')
    plt.xlabel('Training Data Amount (%)')
    plt.ylabel('Test Accuracy')
    plt.title('Training Data Amount vs. Test Accuracy')
    plt.grid(True)
    plt.savefig('data_amount_vs_test_accuracy.png')
    plt.show()

def evaluate_learning_rates(learning_rates):
    results = []

    for lr in learning_rates:
        # Create data generators
        train_generator = datagen.flow(X_train, y_train_cat, batch_size=32)
        test_generator = datagen.flow(X_test, y_test_cat, batch_size=32, shuffle=False)

        # Create and train the model
        model = create_model(learning_rate=lr)
        history = model.fit(train_generator, validation_data=test_generator, epochs=10)

        # Get the final test accuracy
        test_accuracy = history.history['val_accuracy'][-1]
        results.append((lr, test_accuracy))

        print(f"Learning Rate: {lr}, Test Accuracy: {test_accuracy:.2f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, [result[1] for result in results], marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Accuracy')
    plt.title('Learning Rate vs. Test Accuracy')
    plt.grid(True)
    plt.savefig('learning_rate_vs_test_accuracy.png')
    plt.show()

def evaluate_batch_sizes(batch_sizes):
    results = []

    for batch_size in batch_sizes:
        # Create data generators
        train_generator = datagen.flow(X_train, y_train_cat, batch_size=batch_size)
        test_generator = datagen.flow(X_test, y_test_cat, batch_size=32, shuffle=False)

        # Create and train the model
        model = create_model()
        history = model.fit(train_generator, validation_data=test_generator, epochs=10)

        # Get the final test accuracy
        test_accuracy = history.history['val_accuracy'][-1]
        results.append((batch_size, test_accuracy))

        print(f"Batch Size: {batch_size}, Test Accuracy: {test_accuracy:.2f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, [result[1] for result in results], marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy')
    plt.title('Batch Size vs. Test Accuracy')
    plt.grid(True)
    plt.savefig('batch_size_vs_test_accuracy.png')
    plt.show()

def evaluate_dropout_rates(dropout_rates):
    results = []

    for dropout_rate in dropout_rates:
        # Create data generators
        train_generator = datagen.flow(X_train, y_train_cat, batch_size=32)
        test_generator = datagen.flow(X_test, y_test_cat, batch_size=32, shuffle=False)

        # Create and train the model
        model = create_model(dropout_rate=dropout_rate)
        history = model.fit(train_generator, validation_data=test_generator, epochs=10)

        # Get the final test accuracy
        test_accuracy = history.history['val_accuracy'][-1]
        results.append((dropout_rate, test_accuracy))

        print(f"Dropout Rate: {dropout_rate}, Test Accuracy: {test_accuracy:.2f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(dropout_rates, [result[1] for result in results], marker='o')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Test Accuracy')
    plt.title('Dropout Rate vs. Test Accuracy')
    plt.grid(True)
    plt.savefig('dropout_rate_vs_test_accuracy.png')
    plt.show()

def evaluate_dense_units(dense_units_list):
    results = []

    for dense_units in dense_units_list:
        # Create data generators
        train_generator = datagen.flow(X_train, y_train_cat, batch_size=32)
        test_generator = datagen.flow(X_test, y_test_cat, batch_size=32, shuffle=False)

        # Create and train the model
        model = create_model(dense_units=dense_units)
        history = model.fit(train_generator, validation_data=test_generator, epochs=10)

        # Get the final test accuracy
        test_accuracy = history.history['val_accuracy'][-1]
        results.append((dense_units, test_accuracy))

        print(f"Dense Units: {dense_units}, Test Accuracy: {test_accuracy:.2f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(dense_units_list, [result[1] for result in results], marker='o')
    plt.xlabel('Dense Units')
    plt.ylabel('Test Accuracy')
    plt.title('Dense Units vs. Test Accuracy')
    plt.grid(True)
    plt.savefig('dense_units_vs_test_accuracy.png')
    plt.show()

def evaluate_data_augmentation():
    results = []

    # Without data augmentation
    train_generator_no_aug = datagen_without_augmentation.flow(X_train, y_train_cat, batch_size=32)
    test_generator_no_aug = datagen_without_augmentation.flow(X_test, y_test_cat, batch_size=32, shuffle=False)

    model_no_aug = create_model()
    history_no_aug = model_no_aug.fit(train_generator_no_aug, validation_data=test_generator_no_aug, epochs=10)

    test_accuracy_no_aug = history_no_aug.history['val_accuracy'][-1]
    results.append(('No Augmentation', test_accuracy_no_aug))

    print(f"No Augmentation, Test Accuracy: {test_accuracy_no_aug:.2f}")

    # With data augmentation
    train_generator_aug = datagen.flow(X_train, y_train_cat, batch_size=32)
    test_generator_aug = datagen.flow(X_test, y_test_cat, batch_size=32, shuffle=False)

    model_aug = create_model()
    history_aug = model_aug.fit(train_generator_aug, validation_data=test_generator_aug, epochs=10)

    test_accuracy_aug = history_aug.history['val_accuracy'][-1]
    results.append(('With Augmentation', test_accuracy_aug))

    print(f"With Augmentation, Test Accuracy: {test_accuracy_aug:.2f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar([result[0] for result in results], [result[1] for result in results])
    plt.xlabel('Data Augmentation')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Data Augmentation on Test Accuracy')
    plt.grid(True)
    plt.savefig('data_augmentation_vs_test_accuracy.png')
    plt.show()

    # Plot training history
    plot_training_history(history_no_aug, 'without Augmentation')
    plot_training_history(history_aug, 'with Augmentation')

    

# Example usage:
data_amounts = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
# evaluate_data_amounts(data_amounts)

learning_rates = [0.01, 0.001, 0.0001]
# evaluate_learning_rates(learning_rates)

batch_sizes = [32, 64, 128]
# evaluate_batch_sizes(batch_sizes)

dropout_rates = [0.3, 0.5, 0.7]
# evaluate_dropout_rates(dropout_rates)

dense_units_list = [256, 512, 1024]
# evaluate_dense_units(dense_units_list)

evaluate_data_augmentation()
