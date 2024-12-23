import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import cv2

def load_and_preprocess_data(data_dir, img_size=(128, 128)):
    """
    Load and preprocess images from the given directory
    """
    images = []
    labels = []
    
    # Load benign images
    benign_dir = os.path.join(data_dir, 'FNA', 'benign')
    for img_name in os.listdir(benign_dir):
        img_path = os.path.join(benign_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(0)  # 0 for benign
    
    # Load malignant images
    malignant_dir = os.path.join(data_dir, 'FNA', 'malignant')
    for img_name in os.listdir(malignant_dir):
        img_path = os.path.join(malignant_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(1)  # 1 for malignant
    
    return np.array(images), np.array(labels)

def create_model(input_shape):
    """
    Create a CNN model for image classification
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def predict_test_images(model, test_dir, img_size=(128, 128)):
    """
    Predict classes for test images
    """
    predictions = []
    filenames = []
    
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img / 255.0
            pred = model.predict(np.array([img]))[0][0]
            predictions.append("Malignant" if pred > 0.5 else "Benign")
            filenames.append(img_name)
    
    return filenames, predictions

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Parameters
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 20
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data('Dataset2', IMG_SIZE)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile model
    print("Creating model...")
    model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    print("Training model...")
    history = model.fit(X_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_data=(X_val, y_val))
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_val, y_val)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Predict test images
    print("Predicting test images...")
    test_dir = os.path.join('Dataset2', 'test')
    filenames, predictions = predict_test_images(model, test_dir, IMG_SIZE)
    
    # Save predictions to file
    with open('predictions.txt', 'w') as f:
        for filename, prediction in zip(filenames, predictions):
            f.write(f"{filename}: {prediction}\n")
    
    # Save model
    model.save('breast_cancer_model.h5')
    
    print("Processing complete!")

if __name__ == "__main__":
    main()