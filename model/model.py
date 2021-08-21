'''
<<<<<<< HEAD
'''

#imports for tf and resnet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Layer, MaxPooling2D
from keras_adabound import AdaBound

#basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
import re
import seaborn as sn
from pathlib import Path
from datetime import datetime
from fsplit.filesplit import Filesplit
from mlxtend.evaluate import confusion_matrix as cm

#img handling
import PIL
from PIL import Image

#tensorboard functionality
global graph
graph = tf.compat.v1.get_default_graph()

#setting up log directory and tensorboard callbacks
log_dir = Path('./model/logs/')
tbcb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#this is the maximum width and height that ResNet handles.
maxsize = (224, 224)

#specifies the directories for the training and testing.
trainDir = Path('./images/TRAIN')
testDir = Path('./images/TEST')

#this is a list of the 4 classes that will be used. they will be formatted into the directory.
classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

#this is reversed order for the heatmap
rclasses = ['NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE', 'EOSINOPHIL']

def loadImages(trainDir, maxsize, testDir):
    #first step for preprocessing - this uses the preprocessing input imported from the ResNet library.
    trainDGen = ImageDataGenerator(
                preprocessing_function = preprocess_input,
                )

    #fitting images to size as well as specifing the batch size and class mode.
    trainGen = trainDGen.flow_from_directory(
                trainDir,
                target_size = maxsize,
                batch_size = 48, #used to be 32, but since accuracy began to plateau a higher batch would be beneficial.
                class_mode = 'sparse',
                )

    #fitting testing images in a similar format.
    testGen = trainDGen.flow_from_directory(
                testDir,
                target_size = maxsize,
                batch_size = 1,
                class_mode = 'sparse',
                )

    #returns the generated dictionaries
    return trainGen, testGen

def firstModelSetup():
    #model - including the top dense layers for classification, as we are not using ImageNet weights.
    model = ResNet50(
                    include_top = True,
                    weights = None,
                    classes = 4,
                    )

    #compiling the model together. In future there is the possibility of removing the top layer and adding our own dense layers.
    model.compile(
                optimizer = AdaBound(lr=1e-3, final_lr=0.1),
                loss = ['sparse_categorical_crossentropy'],
                metrics = ['accuracy']
                )
    #returns the freshly baked model
    return model

def firstDropOutModelSetup():
    #base model. we are working with resnet as a base, but fine-tuning it with dropout layers.
    bmodel = ResNet50(
                    include_top = False,
                    weights = None,
                    classes = 4,
                    input_shape = (224, 224, 3)
                    )
    #adding our own layers including large amounts of dropout
    hmodel = bmodel.output
    hmodel = AveragePooling2D(pool_size=(7, 7))(hmodel)
    hmodel = Flatten()(hmodel)
    hmodel = BatchNormalization()(hmodel)
    hmodel = Dense(256, activation="relu")(hmodel)
    hmodel = Dropout(0.5)(hmodel)
    hmodel = BatchNormalization()(hmodel)
    hmodel = Dense(128, activation="relu")(hmodel)
    hmodel = Dropout(0.5)(hmodel)
    hmodel = BatchNormalization()(hmodel)
    hmodel = Dense(64, activation="relu")(hmodel)
    hmodel = Dropout(0.5)(hmodel)
    hmodel = BatchNormalization()(hmodel)
    hmodel = Dense(4, activation="softmax")(hmodel)

    #creating the fully fledged model
    model = keras.Model(inputs=bmodel.input, outputs=hmodel)

    #compiling the model together. In future there is the possibility of removing the top layer and adding our own dense layers.
    model.compile(
                optimizer = AdaBound(lr=1e-3, final_lr=0.1),
                loss = ['sparse_categorical_crossentropy'],
                metrics = ['accuracy']
                )

    #returns the freshly baked model
    return model

def trainModel(model, trainGen, epochs):
    global tbcb

    #fitting the model over specified epochs
    model.fit(
            trainGen,
            epochs = epochs,
            callbacks = [tbcb]
            )

    #returns the trained model
    return model

def saveModel(model):
    #if loading model in the same instance as saving
    global savepath
    #creating a path with the exact datetime.
    now = datetime.now()
    savepath = now.strftime("model_%d%m%Y-%H%M")
    os.mkdir("model\saves\{}".format(savepath))

    #THIS IS TOO LARGE TO PUSH TO GITHUB. on its own
    #saving the model in a h5 format
    model.save(
            ("model\saves\{}\model".format(savepath)),
            include_optimizer = True,
            save_format='h5'
            )

    #splitting the file so that it can be pushed
    reSplit(savepath)

    return savepath

def loadModel(savepath):
    #trying if model is in parts
    try:
        #merging the file back
        fs = Filesplit()
        fs.merge(
                "model\saves\{}".format(savepath),
                ("model\saves\{}\model".format(savepath)),
                cleanup = True
                )
    except:
        #if the model is already one file
        pass
    #loading the model at the previous datetime. When Loading, set compile to false due to using a custom optimizer that is not natively recognised.
    model = keras.models.load_model("model\saves\{}\model".format(savepath), compile=False)

    #compiling the model together. In future there is the possibility of removing the top layer and adding our own dense layers.
    model.compile(
                optimizer = AdaBound(lr=1e-3, final_lr=0.1),
                loss = ['sparse_categorical_crossentropy'],
                metrics = ['accuracy']
                )

    #returning the model so it can be used
    return model

def evalModel(model, testGen):
    #evaluating accuracy. verbose set to 2 for silence and information.
    test_loss, test_acc = model.evaluate(
    testGen,
    verbose=2
    )
    print('\nTest accuracy:', test_acc)

    #returns the accuracy value to be compared
    return test_acc


def reSplit(savepath):
    #splitting the file so that it can be pushed
    fs = Filesplit()
    #if errors occur, then it will be caught in the statement
    try:
        fs.split(
            ("model\saves\{}\model".format(savepath)),
            10000000,
            ("model\saves\{}".format(savepath))
            )

        #removing the big file
        os.remove(("model\saves\{}\model".format(savepath)))

        #helpful message showing success
        print("\nModel file split asunder.")
    except:
        print("\nCleaning unsuccessful, try again.")

def reMerge(savepath):
    #merging a file when unable to load
    fs = Filesplit()
    try:
        fs.merge(
            ("model\saves\{}".format(savepath)),
            ("model\saves\{}\model".format(savepath)),
            cleanup = True
            )
    except:
        #if error raised from an improperly saved model
        print("Manifest not found during sweeping.")


#function for creating a confusion matrix and heatmap to easily understand the data and pinpoint the training sections.
def confMatrix(model, testGen, classes):
    #reversing the classes if function used for another dataset
    rclasses = classes[::-1]
    #predicting the test set
    predicts = model.predict(testGen)
    yPredicted = np.argmax(predicts, axis=1)
    #looping the testGen to obtain the true labels, then setting it as an array
    test_labels = []

    for i in range(0, testGen.samples):
        test_labels.extend(np.array(testGen[i][1]))

    yActual = np.asarray(test_labels, dtype=int)

    #setting up a matrix using the mlxtend library
    matrix = cm(yActual, yPredicted)

    #plotting the heatmap using seaborn
    #red colour, annotating the cells with their values and annotating the x and y classes with their variables.
    sn.heatmap(matrix, cmap='Reds', annot=True, fmt='d', xticklabels = classes, yticklabels = classes)

#function for plotting an accuracy graph based upon the previous models' results (completed after all training has occurred)
def plotAccuracy():

    #creating a list to store the accuracies
    modelAcc = [0.62002414, 0.71250504, 0.80458385, 0.86087656, 0.8954564, 0.87856853, 0.8713309]
    dmodelAcc = [0.46441495, 0.8065943, 0.7599518, 0.8194612, 0.8725372, 0.89103335, 0.9010334]

    modelAccs = [[0.62002414, 0.46441495], [0.71250504, 0.8065943], [0.80458385, 0.7599518], [0.86087656, 0.8194612], [0.8954564, 0.8725372], [0.87856853, 0.89103335], [0.8713309,  0.9010334]]

    #creating the dataframe
    df = pd.DataFrame(columns=['modelAcc', 'dmodelAcc'], data=modelAccs)

    #plotting
    sn.relplot(kind="line", data=df, palette = "BuGn")

'''
=======
hello
>>>>>>> parent of 3e79063 (first training + model complete)
'''
