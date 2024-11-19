import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import optimizers
import os
import cv2

def make_square(path):
    ''' Reize the image to 256x256 dimension '''
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    cv2.imwrite('media/image.png', image)

def load_image(image_path):
    ''' Return the image in the format required by VGG16 model. '''
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_features(feature_extractor, image):
    ''' Returns the features extracted by the model. '''
    return feature_extractor.predict(load_image(image))

def cosine_similarity_fn(anchor_image_feature, test_image_feature):
    ''' Returns the features extracted by the model. '''
    return cosine_similarity(anchor_image_feature, test_image_feature)[0][0]


def verify(anchor_image, gan_op):

    feature_extractor = tf.keras.models.load_model(
        r"/Users/akhilbabu/Documents/work/Signature-Verification/SOURCE/vgg_finetuned_model/bbox_regression_cnn.h5",
        compile=False
    )

    # Compile the model with a default optimizer for inference, if needed
    feature_extractor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                              loss='categorical_crossentropy', 
                              metrics=['accuracy'])
    
    
    feature_set = []
    # anchor image is resized to 256x256 to match outputs from gan.
    make_square(anchor_image)
    anchor_image_feature = extract_features(feature_extractor, anchor_image)
    # test_images = [gan_op + image for image in os.listdir(gan_op) if image[2:6]=='fake']
        # Use os.path.join to construct file paths correctly
    test_images = [
        os.path.join(gan_op, image)
        for image in os.listdir(gan_op)
        if image[2:6] == 'fake'
    ]
    for image in test_images:
        test_image_feature = extract_features(feature_extractor, image)
        cosine_similarity = cosine_similarity_fn(anchor_image_feature, test_image_feature)
        cosine_similarity = round(cosine_similarity, 2)
        feature_set.append([image, cosine_similarity])
    return feature_set
