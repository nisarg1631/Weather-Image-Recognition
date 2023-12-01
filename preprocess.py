import os
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave

# open images from dataset directory and resize them to 200,200,3

dataset_path = 'dataset/'
resized_path = 'resized/'
classes = []

# create resized directory if it doesn't exist
if not os.path.exists(resized_path):
    os.mkdir(resized_path)

    # create subdirectories in resized directory for each class
    for subdir in os.listdir(dataset_path):
        if not os.path.exists(resized_path + subdir):
            os.mkdir(resized_path + subdir)

# subdirectories in dataset directory are classes
for subdir in os.listdir(dataset_path):
    for file in os.listdir(dataset_path + subdir):
        if not os.path.exists(resized_path + subdir + '/' + file):
            try:
                img = imread(dataset_path + subdir + '/' + file)
                img = resize(img, (200, 200, 3))
                img = np.array(img * 255, dtype=np.uint8)
                imsave(resized_path + subdir + '/' + file, img)
            except Exception as e:
                print('Error in file: ' + file)
    print(f'Class {subdir} loaded.')
