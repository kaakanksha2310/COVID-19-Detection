import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt


# Path to dataset
dataset_dir =  "/Users/aakankshakumari/Desktop/Project/CovidXrayDetection/COVID-19_Radiography_Dataset"
categories = ["COVID", "Normal", "Viral Pneumonia"]
img_size = 224  # Resize images to 224x224
valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]  # List of valid image file extensions

def load_images():
    data = []
    labels = []
    for category in categories:
        # Adjust the path to look into the "images" folder inside each category
        path = os.path.join(dataset_dir, category, "images")
        class_num = categories.index(category)  # Assign numerical labels (0 for COVID, 1 for Normal, etc.)
        
        # Ensure the "images" folder exists
        if not os.path.exists(path):
            print(f"Images folder not found: {path}, skipping.")
            continue
        
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            # Check if the file has a valid image extension
            if os.path.splitext(img)[1].lower() in valid_image_extensions:
                try:
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_array is None:
                        print(f"Failed to load image {img_path}, skipping.")
                        continue
                    resized_img = cv2.resize(img_array, (img_size, img_size))
                    data.append(resized_img)
                    labels.append(class_num)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
            else:
                print(f"Skipping non-image file: {img_path}")
    
    return np.array(data), np.array(labels)

# Load and preprocess images
X, y = load_images()

# Check if data is loaded
if len(X) == 0 or len(y) == 0:
    raise Exception("No valid images were loaded. Please check the dataset path and image files.")


if len(X) == 0 or len(y) == 0:
    print("No data loaded. Please check the dataset.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Normalize the data
X = X / 255.0  # Scale pixel values to between 0 and 1
X = X.reshape(-1, img_size, img_size, 1)  # Add a channel dimension for grayscale images

# One-hot encode labels
y = to_categorical(y, len(categories))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(categories), activation='softmax'))  # Output layer for classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

predictions = model.predict(X_test)
for i in range(5):
    plt.imshow(X_test[i].reshape(img_size, img_size), cmap='gray')
    plt.title(f"Prediction: {categories[np.argmax(predictions[i])]}, Actual: {categories[np.argmax(y_test[i])]}")
    plt.show()
