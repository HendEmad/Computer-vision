from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input  # For input preprocessing
from keras.preprocessing import image  # The input is an image
from keras.preprocessing.image import ImageDataGenerator  # For image augmentation
from keras.models import Sequential  # Layers, for sequential models like VGG16
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# Re-size the images to the size of images that VGG16 is trained on
IMAGE_SIZE = [224, 224]

train_path = 'DataSet/Train'
test_path = 'DataSet/Test'

# Add preprocessing layer to the front of VGG
# Remove the last layer, as it includes 1000 classes the model has been trained on
# [3] is number of channels (RGB), in case a black and white images, it will be [1]
# The second parameter is default one in keras
# The last parameter is to tell if the last layer needs to be added in our VGG16 model or not
# If the last parameter is assigned to be `False`, then we can obtain transfer learning
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# We don't have to train the model layers again, as it has already trained, the weights are fixed; as it is a state-of-art algorithm.
# If we tried to re-train the model layer, it will give us a worse accuracy, as this VGG16 has been trained in many images where a lot of GPU Power and resources are required.
# Make sure that for each and every layer in VGG16
# `trainable` ==> is a parameter which will tell that we have to train the layers or not
for layer in vgg.layers:
    layer.trainable = False

# In order to add the last layer
folders = glob('DataSet/Train/*')  # This will tell no. of categories(folders) are inside the train dataset
# print(folders)  # ['DataSet/Train\\Elon', 'DataSet/Train\\Hend', 'DataSet/Train\\Salma', 'DataSet/Train\\Sisi']

# Flatten the last layer of VGG16
x = Flatten()(vgg.output)

# Appending my train dataset folders in as a dense layer with an activation function softmax with x value
# This means that this particular code will add the folders which actually means the no. of categories at the last layer
prediction = Dense(len(folders), activation='softmax')(x)

# Converting this prediction into a model
model = Model(inputs=vgg.input, outputs=prediction)

# View the structure of the model
model.summary()

# Compilation step
# Tell the model what cost and optimization method to use
model.compile(
    loss = 'categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Image augmentation step
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'DataSet/Train', target_size=(224, 224), batch_size=32, class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'DataSet/Test', target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# Fit the model step
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# Accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf
from keras.models import load_model
model.save('faceFeatures_newmodel.h5')