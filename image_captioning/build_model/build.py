"""
This file contains the code for building and evaluating the deep-learning image captioning model, based on the Flickr8K dataset.
Steps:
1. Extract features from images using pre-trained CNN (here I use VGG-16)
2. Pre-process text data
3. Build the deep-learning model (my model is based on the merge model as described by Tanti, et al. (2017). Where to put the Image in an Image Caption Generator.)
4. Progressive model training (since the memory of my computer is insufficient for loading the entire dataset at once)
5. Evaluation based on BLEU score
"""

import os
import tensorflow as tf
import keras
import numpy as np

# 1. Extract image features using pre-trained CNN.

def feature_extractions(directory):
    """
    Input: directory of images
    Return: A dictionary of features extracted by VGG-16, size 4096.
    """
    
    model = tf.keras.applications.vgg16.VGG16()
    model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output) #Remove the final layer
    
    features = {}
    for f in os.listdir(directory):
        filename = directory + "/" + f
        identifier = f.split('.')[0]
        
        image = tf.keras.utils.load_img(filename, target_size=(224,224))
        arr = tf.keras.utils.img_to_array(image, dtype=np.float32)
        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        arr = keras.applications.vgg16.preprocess_input(arr)
    
        feature = model.predict(arr, verbose=0)
        features[identifier] = feature
        
        print("feature extraction: {}".format(f))
    return(features)

def sample_caption(model, tokenizer, max_length, vocab_size, feature):
    """
    Input: model, photo feature: shape=[1,4096]
    Return: A generated caption of that photo feature. Remove the startseq and endseq token.
    """
    
    caption = "<startseq>"
    while 1:
        #Prepare input to model
        encoded = tokenizer.texts_to_sequences([caption])[0]
        padded = tf.keras.utils.pad_sequences([encoded], maxlen=max_length, padding='pre')[0]
        padded = padded.reshape((1, max_length))
        
        pred_Y = model.predict([feature, padded])[0,-1,:]
        next_word = tokenizer.index_word[pred_Y.argmax()]
        
        #Update caption
        caption = caption + ' ' + next_word
        
        #Terminate condition: caption length reaches maximum / reach endseq
        if next_word == '<endseq>' or len(caption.split()) >= max_length:
            break
    
    #Remove the (startseq, endseq)
    caption = caption.replace('<startseq> ', '')
    caption = caption.replace(' <endseq>', '')
    
    return(caption)
