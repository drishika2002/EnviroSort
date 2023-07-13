#!/usr/bin/env python
# coding: utf-8
# In[28]:

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[29]:

classifier = Classifier('static/Resources/Model/keras_model.h5', 'static/Resources/Model/labels.txt')


# ## DATA VISUALIZATION

# In[4]:


frequencies = []
categories = []
pathFolderBins = 'static/Resources/dataset'
pathList = os.listdir(pathFolderBins)
for path in pathList:
    categories.append(path)
# print(categories)

for category in categories:
    folder_path = os.path.join('static/Resources/dataset/', category)
    file_count = len(os.listdir(folder_path))
    frequencies.append(file_count)

bar_width = 0.7
fig, ax = plt.subplots(figsize=(14, 5))

ax.bar(categories, frequencies, width=bar_width)
ax.set_xlabel('Waste Categories')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Waste Categories')
plt.show()


# In[5]:


plt.pie(frequencies, labels=categories, autopct='%1.1f%%')
plt.title('Proportion of Waste Categories')
plt.show()


# In[6]:


thumbnail_size = (60, 70)

num_categories = len(categories)
num_rows = int((num_categories + 1) / 2)
num_cols = 4 if num_categories > 1 else 1

fig = plt.figure(figsize=(8, 8))

for i, category in enumerate(categories):
    category_path = os.path.join('static/Resources/dataset', category)
    image_filenames = os.listdir(category_path)

    selected_image = image_filenames[2]

    image_path = os.path.join(category_path, selected_image)
    image = Image.open(image_path)
    image.thumbnail(thumbnail_size)

    ax = fig.add_subplot(num_rows, num_cols, i+1)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(category)

plt.tight_layout()
plt.show()
 
# ### Waste categories(Labels):
# 
# 0 = White glass
# 1 = Green glass
# 2 = Brown glass
# 3 = Food items
# 4 = Plastic
# 5 = Trash
# 6 = Metal
# 7 = Paper
# 8 = Cardboard
# 9 = Battery
# 10 = Shoes
# 11 = Clothes

# In[7]:


def process_image(image):
    imgS = cv2.resize(image, (175, 175))

    imgBackground = cv2.imread('static/Resources/background.png')
    imgBackground = cv2.resize(imgBackground, (1000, 750))

    prediction = classifier.getPrediction(imgS, pos=(20,20),color=(0,255,100),scale=0.5)
    print(prediction)
    
    imgArrow = cv2.imread('static/Resources/arrow.png', cv2.IMREAD_UNCHANGED)
    imgArrow = cv2.resize(imgArrow, (170, 90))
    
    label_id = prediction[1]
    
    # label order is different than categories list order
    bin_mapping = {
        0: 3,
        1: 3,
        2: 3,
        3: 5,
        4: 7,
        5: 7,
        6: 4,
        7: 6,
        8: 6,
        9: 0,
        10: 1,
        11: 1
    }
    
    bin_index = bin_mapping.get(label_id, -1)
    
    imgBinsList = []
    pathFolderBins = 'static/Resources/bins'
    pathList = os.listdir(pathFolderBins)
    for path in pathList:
        imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

    imgBinsList[bin_index] = cv2.resize(imgBinsList[bin_index], (175, 200))

    imgBackground[100:100+175, 150:150+175] = imgS
    imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (425, 150))
    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[bin_index], (600, 100))

    cv2.imshow("Output", imgBackground)



print("Class Counts:")
for class_id, count in enumerate(frequencies):
    print(f"Class ID: {class_id}, Count: {count}")


# In[35]:


import random
import matplotlib.pyplot as plt

percentage_of_majority = 0.7
target_samples_per_class = int(max(frequencies) * percentage_of_majority)

aug_samples = []
minority_classes = []

aug_categories = []



for class_id, count in enumerate(frequencies):
    
    if count < target_samples_per_class:
        num_aug = target_samples_per_class - count
        minority_classes.append(class_id)
        aug_samples.append(num_aug)
        aug_categories.append(categories[class_id])

print(aug_categories)
print("Minority Classes:")
for class_id in minority_classes:
    print(f"Class ID: {class_id}")

print("Number of Augmented Samples:")
for class_id, num_aug in zip(minority_classes, aug_samples):
    print(f"Class ID: {class_id}, Augmentations: {num_aug}")

# Plotting the bar chart
plt.figure(figsize=(8, 6))
plt.bar(range(len(minority_classes)), aug_samples)
plt.xlabel('Waste Categories')
plt.ylabel('Number of Augmented Samples')
plt.title('Original + Augmented Dataset')
plt.xticks(range(len(aug_categories)), aug_categories)
plt.show()

# Plotting the pie chart
plt.figure(figsize=(6, 6))
plt.pie(aug_samples, labels=aug_categories, autopct='%1.1f%%')
plt.title('Original + Augmented Dataset')
plt.show()


# In[22]:


def augment_minority_classes(class_id, image, num_aug):
    aug_images = []
    
    for _ in range(num_aug):
        aug_image = datagen.random_transform(image)
        aug_images.append((class_id, aug_image))

    return aug_images


# In[37]:


import random
augmented_cnts = [0] * len(frequencies)

def capture_from_camera():
    cap = cv2.VideoCapture(0)
    while True:
        frame_cap, img = cap.read()
        if not frame_cap: 
            print("Failed to capture frame.")
            break
        process_image(img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def process_single_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read the image file.")
    else:
        process_image(img)
        cv2.waitKey(0)
        

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# j = 0
# dataset_path = 'static/Resources/dataset'
# subdirectories = os.listdir(dataset_path)
# for subdir in subdirectories:
#     subdir_path = os.path.join(dataset_path, subdir)
#     image_files = os.listdir(subdir_path)
#     random.shuffle(image_files)

#     class_id = minority_classes[j]
#     j += 1
#     count = len(image_files)

#     if count < target_samples_per_class:
#         num_aug = target_samples_per_class - count
#         augmented_samples = augment_minority_classes(class_id, img, num_aug)
#         augmented_images.extend(augmented_samples)
#         augmented_cnts[class_id] += len(augmented_samples)

#     for image_file in image_files:
#         image_path = os.path.join(subdir_path, image_file)
#         img = cv2.imread(image_path)
#         if img is None:
#             print("Failed to read the image file: ", image_path)
#         else:
#             process_image(img)


# In[ ]:





# In[38]:


option = input("Choose:\n1. Camera Capture\n2. Image Upload\n")
if option == '1':
    capture_from_camera()
else:
    image_path = input("Enter the path to the image file: ")
    process_single_image(image_path) 

cv2.destroyAllWindows()


# In[ ]:




