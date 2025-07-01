# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 20:05:36 2025

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
import cv2
import matplotlib.image as mpimg
import seaborn as sns
import zipfile

#%matplotlib inline
plt.style.use('ggplot')

# I am adding comments so I remember what is what (basically inline documentation)
# Adding definitions and reasonings also so that later we dont need to search on internet/gpt
# I wish github had a feature where there were like 2 panes so that you could document side by side and line by line instead of comments :'))

#-----------------------------------------------------------------------------------------------------
#Dataset + unzipping files
'''z = zipfile.ZipFile('archive.zip')

z.extractall()'''

# remember to comment this part out after you've run this once, or it will keep extracting the zipfile

#-----------------------------------------------------------------------------------------------------
#File renaming (Data Preprocessing): 
    
'''folder = 'brain_tumor_dataset/yes/'
count = 1

for filename in os.listdir(folder):
    source = folder + filename
    destination = folder + "Y_" +str(count)+".jpg"
    os.rename(source, destination)
    count+=1
print("All files are renamed in the yes dir.")

folder = 'brain_tumor_dataset/no/'
count = 1

for filename in os.listdir(folder):
    source = folder + filename
    destination = folder + "N_" +str(count)+".jpg"
    os.rename(source, destination)
    count+=1
print("All files are renamed in the no dir.")'''

# remember to comment this part out after you've run this once, or it will say that the files are already named as such


#-----------------------------------------------------------------------------------------------------
# Exploratory Data Analysis(EDA):
    
listyes = os.listdir("brain_tumor_dataset/yes/")
number_files_yes = len(listyes)
print("No. of positive samples: ", number_files_yes) #no. of yes images

listno = os.listdir("brain_tumor_dataset/no/")
number_files_no = len(listno)
print("No. of negative samples: ", number_files_no) #no. of no images

#-----------------------------------------------------------------------------------------------------
# Plot of Data

data = {'tumorous': number_files_yes, 'non-tumorous': number_files_no}

typex = data.keys()
values = data.values()

fig = plt.figure(figsize=(5,7))
plt.bar(typex, values, color="blue")

plt.xlabel("Type")
plt.ylabel("No of Brain MRI Scans")
plt.title("Count of Brain Tumor Images")
plt.show()

#-----------------------------------------------------------------------------------------------------
# Data Augumentation(DA):
    
# DA: Artificially increasing the size and diversity of your training dataset by creating modified versions of existing data.
# Why? Prevents overfitting by exposing the model to varied data, improves generalization to unseen data, reduces need for collecting more data, which is often expensive or difficult
# How? Rotation, Flipping, Scaling / Zooming, Translation, Brightness/Contrast, Noise Injection, Shearing, Cropping, Elastic distortion
# How in Python? TensorFlow/Keras: ImageDataGenerator, tf.image, tf.keras.layers.Random

#Here the data from kaggle has 155 yes & 98 no images
# Therefore 155(61%), 98(39%) and so imbalance, so we do this now:
    
import tensorflow as tf
#an open-source machine learning library developed by Google to build, train, and deploy ML and DL models.
#Keras is a high-level API that runs on top of TensorFlow to make it easier to use

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
#to augment image data (like rotation, flip, zoom, etc.)

from tensorflow.keras.models import Model  
#to define a custom model architecture by modifying or extending an existing base model (like VGG19) using the Model class.

from tensorflow.keras.layers import Flatten, Dense, Dropout 
# Flatten: Flattens multidimensional output into a 1D vector
# Dense: Fully connected (neural network) layer
# Dropout: Randomly turns off a portion of neurons during training to prevent overfitting

from tensorflow.keras.applications.vgg19 import VGG19
#VGG19 is a deep convolutional neural network (CNN) architecture with 19 layers (16 convolutional + 3 fully connected) developed by the Visual Geometry Group at Oxford.
#It is used for Transfer Learning: Instead of training a CNN from scratch (which needs a lot of data), we: Reuse the lower layers of a pre-trained model (like VGG19) and add a new classifier layer on top to suit our specific task (like tumor vs. no tumor).

from tensorflow.keras.optimizers import SGD, Adam  
#optimizers to update model weights during training:
#SGD: Stochastic Gradient Descent #Adam: Adaptive optimizer 

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#training callbacks that automate important behaviors:
#ModelCheckpoint: Saves the model to disk when validation accuracy improves
#EarlyStopping: Stops training if the model stops improving (prevents overfitting)
#ReduceLROnPlateau: Reduces the learning rate if the model's performance plateaus


#REMEMBER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
#COMMENT THIS PART OUT AFTER 1 TIME AUGMENTATION!!! or it will keep creating augmented images everytime you run the code
#if you want to run the code multiple times, just clear out the augmented/yes & /no folders before
#although even if you do multiple augmentations the % of yes and no remain same, so chill scene there:)

# to convert seconds into h:m:s time format
'''def timing(sec_elapsed):
    h = int(sec_elapsed / (60*60))
    m = int(sec_elapsed % (60*60) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{s}"

# to generate augmented images to increase dataset:
# n_generated_samples: How many augmented images to generate per original image

def augmented_data(file_dir, n_generated_samples, save_to_dir):
    data_gen = ImageDataGenerator(rotation_range=10,   # data_gen object that can apply random augmentations to images.
                      width_shift_range=0.1,
                      height_shift_range=0.1,
                      shear_range=0.1,
                      brightness_range=(0.3, 1.0),
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='nearest')
    
    for filename in os.listdir(file_dir):     #loops through images, loads image using open cv, reshapes it as needed by kera's .flow() 
        image = cv2.imread(file_dir + '/' + filename)
        image = image.reshape((1,) + image.shape)
        save_prefix = 'aug_' + filename[:-4]
        i=0
        for batch in data_gen.flow(x = image, batch_size = 1, save_to_dir = save_to_dir, save_prefix = save_prefix, save_format = "jpg"):
            i+=1
            if i>n_generated_samples:
                break


import time
start_time = time.time()  #to calculate the time taken for image augmentations

yes_path = 'brain_tumor_dataset/yes' 
no_path = 'brain_tumor_dataset/no'
augmented_data_path = 'augmented_data/'  #to store augmented images

augmented_data(file_dir = yes_path, n_generated_samples=6, save_to_dir=augmented_data_path+'yes')
augmented_data(file_dir = no_path, n_generated_samples=9, save_to_dir=augmented_data_path+'no')

end_time = time.time()
execution_time = end_time - start_time
print(timing(execution_time))


# summary of data augmentation
def data_summary(main_path):
    yes_path = "augmented_data/yes/" 
    no_path = "augmented_data/no/"
    
    n_pos = len(os.listdir(yes_path))  #no. of yes augmented images
    n_neg = len(os.listdir(no_path))   #no. of no augmented images
    
    n = (n_pos + n_neg)  #total no. of augmented images generated
     
    pos_per = (n_pos*100)/n  #yes augmented %
    neg_per = (n_neg*100)/n  #no augmented %
    
    print(f"Number of sample: {n}")
    print(f"{n_pos} Number of positive sample in percentage: {pos_per}%")
    print(f"{n_neg} Number of negative sample in percentage: {neg_per}%")
    
data_summary(augmented_data_path)'''


#Now we generate the same data plot as above, but just for the aumented images instead:
    
#EDA:
listyes = os.listdir("augmented_data/yes/")
number_files_yes = len(listyes)
print("No. of positive augmented images: ", number_files_yes)

listno = os.listdir("augmented_data/no/")
number_files_no = len(listno)
print("No. of negative augmented images: ", number_files_no)

#Plot:
data = {'tumorous': number_files_yes, 'non-tumorous': number_files_no}

typex = data.keys()
values = data.values()

fig = plt.figure(figsize=(5,7))
plt.bar(typex, values, color="lightgreen")

plt.xlabel("Type")
plt.ylabel("No of Brain Tumor Images")
plt.title("Count of Augmented Brain Tumor Images")
plt.show() #Compare the 2 plots, this one has less imbalance

#-----------------------------------------------------------------------------------------------------

# Data Preprocessing(DP):
    
# The augmented images have a lot of extra unnecessary background pixels, in DP we remove that
# We do this by identifying the contours (borders) of the brain in the images & then cropping the image
# DP Steps: 1. Convert BGR TO GRAY | 2. GaussianBlur | 3. Threshold | 4. Erode | 5. Dilate | 6. Find Contours

#imutils: a small utility library built on top of OpenCV (cv2) to make common image processing tasks simpler and shorter to write.
#imutils automatically extracts the contour list from whatever format cv2.findContours() returns and makes your code independent of cv2 version

import imutils

def crop_brain_tumor(image, plot=False):      
#Takes an MRI image as input and processes it to find and crop the brain region. If plot=True, shows the original and cropped image side by side

#MRI images are often colored in BGR (Blue, Green, Red) when loaded by OpenCV. But for most image processing tasks, especially thresholding, we work in grayscale.
#Applying Gaussian Blur smooths the image and reduces noise, making it easier to isolate large structures like the brain.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #1. Convert BGR TO GRAY
    gray = cv2.GaussianBlur(gray, (5,5), 0)  #2. GaussianBlur

#Thresholding the image converts the grayscale image to binary: 
#Pixels with intensity > 45 become white (255). All others become black (0). This helps in identifying the bright regions like the brain. 
    thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]   #3. Threshold the Image
    
#Erosion and Dilation: These are morphological operations. They help in refining the thresholded mask to better represent the brain contour.
# Erosion: Shrinks white regions (removes small blobs/noise)
# Dilation: Expands them back (restores shape)
    thres =cv2.erode(thres, None, iterations = 2)   #4. Erode
    thres = cv2.dilate(thres, None, iterations = 2) #5. Dilate

#Find External Contours: This finds the outlines of white blobs (potential brain regions).
# RETR_EXTERNAL means we only care about the outermost shape.
# grab_contours() standardizes the output to work across OpenCV versions.
    cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)   #6. Find Contours
    c = max(cnts, key = cv2.contourArea) #Get the Largest Contour- Assumes the largest blob is the brain.


#Find Extreme Points: This finds the bounding rectangle of the contour:    
    extLeft = tuple(c[c[:,:,0].argmin()][0])   #leftmost point (minimum x)
    extRight = tuple(c[c[:,:,0].argmax()][0])  #rightmost point (maximum x)
    extTop = tuple(c[c[:,:,1].argmin()][0])    #topmost point (minimum y)
    extBot = tuple(c[c[:,:,1].argmax()][0])    #bottommost point (maximum y)

# Crop the Region of Interest using the above extreme points:    
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]] 

#Plot the original vs cropped image:
    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        
        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)  # hides the axes
        
        plt.title('Original Image')
            
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)  # hides the axes

        plt.title('Cropped Image')
        plt.show()
    return new_image

#Example Calls:
img = cv2.imread('augmented_data/yes/aug_Y_1_0_330.jpg') 
crop_brain_tumor(img, True)

img = cv2.imread('augmented_data/no/aug_N_1_0_80.jpg')
crop_brain_tumor(img, True)

#-----------------------------------------------------------------------------------------------------

#Image Loading & Visualisation

# Shuffle: randomly shuffles your dataset — keeping the input X and labels y aligned.
# Why? When you load your images (like from folders named “yes” and “no”), they’re likely grouped — all the “yes” images, then all the “no” ones.
# If you don’t shuffle, your model might only see tumor images first and learn a biased pattern. It could also fail on validation data if the distribution is different.

# Interpolation: When you resize an image, especially increase or decrease its dimensions, you end up with missing pixels (when enlarging) or extra pixels (when shrinking).
# Interpolation is the method used to estimate the values of those new pixels.

from sklearn.utils import shuffle

def load_data(dir_list, image_size):
    X=[]  # stores image data
    y=[]  ## stores corresponding labels (yes/no tumor)
    
    image_width, image_height=image_size  #a tuple
    
    for directory in dir_list:                     #For every image inside both folders
        for filename in os.listdir(directory):
            
            image = cv2.imread(directory + '/' + filename)  #read image using open cv
            image = crop_brain_tumor(image, plot=False)     #crop brain image, dont plot it 
            
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation = cv2.INTER_CUBIC)  #Resizes image to 240x240 using cubic interpolation (good quality)
            image = image/255.00   #Divides pixel values by 255 to Normalize to [0, 1]
            
            X.append(image)   #Adds the image to list X
            if directory[-3:] == "yes":   #Adds label to list y: 1 if image came from folder ending in "yes"; 0 if from "no"
                y.append(1)
            else:
                y.append(0)
                
    X=np.array(X)  #Converts lists to NumPy arrays (needed for ML models)
    y=np.array(y)
    X,y = shuffle(X,y)  #Shuffles the dataset to randomize tumor/no-tumor order
    
    print(f"Number of examples: {len(X)}")
    print(f"X SHAPE is : {X.shape}")   #X shape: (No.of images(2065), size of each image(240,240), each image has 3 channels(RGB))->(2065,240,240,3)
    print(f"y SHAPE is : {y.shape}")   #y shape: (No. of labels to images, )->(2065, )
    return X,y

# Load Data From Directories
augmented_path = 'augmented_data/'
augmeneted_yes = augmented_path + 'yes'
augmeneted_no = augmented_path + 'no'

IMAGE_WIDTH, IMAGE_HEIGHT = (240,240)

X,y = load_data([augmeneted_yes, augmeneted_no], (IMAGE_WIDTH, IMAGE_HEIGHT)) #Calls the function and stores preprocessed images in X, and labels in y.

# Visualize Sample Images: Function to show n sample images per class (tumor = 1, no tumor = 0)
def plot_sample_images(X, y, n=50):

    for label in [0,1]:
        images = X[np.argwhere(y == label)]  #This returns the array indices where y equals the label & then stores the images at those indices
        
        # Plot the images: Takes only n images from that class & arranges them in a 10-column grid
        n_images = images[:n]
        columns_n = 10
        rows_n = int(n/ columns_n)
        plt.figure(figsize=(20, 10))
        
        i = 1        
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)  #Add subplots (adds each image in the the image grid)
            plt.imshow(image[0])  #[0] to removes the extra array wrapper/ dimension caused by argwhere
            
            plt.tick_params(axis='both', which='both', 
                            top=False, bottom=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False,
                            labelright=False)   #hides axis ticks
            i += 1
        
        label_to_str = lambda label: "Yes" if label == 1 else "No" 
        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")  #Adds title based on tumor status
        plt.show()

plot_sample_images(X,y)

#-----------------------------------------------------------------------------------------------------

# Data Spliting: Creating Train, Test, Validation datasets

#Creating folders, can be done manually too:
if not os.path.isdir('tumorous_and_nontumorous'):
    base_dir = 'tumorous_and_nontumorous'
    os.mkdir(base_dir)
    
if not os.path.isdir('tumorous_and_nontumorous/train'):
    train_dir = os.path.join(base_dir , 'train')
    os.mkdir(train_dir)
if not os.path.isdir('tumorous_and_nontumorous/test'):
    test_dir = os.path.join(base_dir , 'test')
    os.mkdir(test_dir)
if not os.path.isdir('tumorous_and_nontumorous/valid'):
    valid_dir = os.path.join(base_dir , 'valid')
    os.mkdir(valid_dir)
    
if not os.path.isdir('tumorous_and_nontumorous/train/tumorous'):
    infected_train_dir = os.path.join(train_dir, 'tumorous')
    os.mkdir(infected_train_dir)
if not os.path.isdir('tumorous_and_nontumorous/test/tumorous'):
    infected_test_dir = os.path.join(test_dir, 'tumorous')
    os.mkdir(infected_test_dir)
if not os.path.isdir('tumorous_and_nontumorous/valid/tumorous'):
    infected_valid_dir = os.path.join(valid_dir, 'tumorous')
    os.mkdir(infected_valid_dir)

if not os.path.isdir('tumorous_and_nontumorous/train/nontumorous'):
    healthy_train_dir = os.path.join(train_dir, 'nontumorous')
    os.mkdir(healthy_train_dir)
if not os.path.isdir('tumorous_and_nontumorous/test/nontumorous'):
    healthy_test_dir = os.path.join(test_dir, 'nontumorous')
    os.mkdir(healthy_test_dir)
if not os.path.isdir('tumorous_and_nontumorous/valid/nontumorous'):
    healthy_valid_dir = os.path.join(valid_dir, 'nontumorous')
    os.mkdir(healthy_valid_dir) 
#-----------------------------------------------------------------------------------------------------