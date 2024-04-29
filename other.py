import os
import cv2
import numpy as np


def preprocess_data(project_dir):
    images_dir = os.path.join(project_dir, 'images')

    images = []

    for image_name in os.listdir(images_dir):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(images_dir, image_name)

            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))  # Resize image to match FPN input size
            image = image / 255.0

            images.append(image)

    return np.array(images)


# Usage example:
project_dir = r'C:\Users\Admin\PycharmProjects\fashion'
images = preprocess_data(project_dir)
