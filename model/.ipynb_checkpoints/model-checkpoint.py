<<<<<<< HEAD
#imports for tf and resnet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras_adabound import AdaBound

#basic imports
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re
import seaborn as sn
from pathlib import Path
from datetime import datetime
from fsplit.filesplit import Filesplit
from mlxtend.plotting import confusion_matrix as cm


#img handling
import PIL
from PIL import Image

#this is the maximum width and height that ResNet handles.
maxsize = (224, 224)

#specifies the directories for the training and testing.
trainDir = Path('./images/TRAIN')
testDir = Path('./images/TEST_SIMPLE')

#this is a list of the 4 classes that will be used. they will be formatted into the directory.
classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

#this is reversed order for the heatmap
rclasses = ['NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE', 'EOSINOPHIL']

#first step for preprocessing - this uses the preprocessing input imported from the ResNet library.
trainDGen = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            )

#fitting images to size as well as specifing the batch size and class mode.
trainGen = trainDGen.flow_from_directory(
            trainDir,
            target_size = maxsize,
            batch_size = 32,
            class_mode = 'sparse',
            )

#fitting testing images in a similar format.
testGen = trainDGen.flow_from_directory(
            testDir,
            target_size = maxsize,
            batch_size = 1,
            class_mode = 'sparse',
            )

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

model.fit(
        trainGen,
        epochs = 10
        )

#creating a path with the exact datetime
now = datetime.now()
savepath = now.strftime("model_%d%m%Y-%H%M")
os.mkdir("model\saves\{}".format(savepath))

#THIS IS TOO LARGE TO PUSH TO GITHUB. on its own
#saving the model
model.save(
        ("model\saves\{}\model".format(savepath)),
        include_optimizer = True,
        save_format='h5'
        )

#splitting the file so that it can be pushed
fs = Filesplit()
fs.split(
        ("model\saves\{}\model".format(savepath)),
        10000000,
        ("model\saves\{}".format(savepath))
        )

#removing the big file
os.remove(("model\saves\{}\model".format(savepath)))

#merging the file back
fs.merge(
        "model\saves\{}".format(savepath),
        ("model\saves\{}\model".format(savepath)),
        cleanup = True
        )

'''
#save model as json
modJson = model.to_json()

#creating a path with the exact datetime
now = datetime.now()
savepath = now.strftime("model_%d%m%Y-%H%M")
os.mkdir("model\saves\{}".format(savepath))

with open(("model\saves\{}\config.json".format(savepath)), "wt") as f:
    f.write(modJson)
    model.save_weights("model\saves\{}\weights.hd5".format(savepath))
f.close()
'''



#evaluating accuracy. verbose set to 2 for silence and information.
test_loss, test_acc = model.evaluate(testGen, verbose=2)
print('\nTest accuracy:', test_acc)

#loading the model at the previous datetime. When Loading, set compile to false due to using a custom optimizer that is not natively recognised.
model2 = keras.models.load_model("model\saves\{}\model".format(savepath), compile=False)

#predicting the test set
predicts = model.predict(testGen)
yActual = np.asarray(test_labels, dtype=int)

#looping the testGen to obtain labels, then setting it as an array
test_labels = []

for i in range(0, 71):
    test_labels.extend(np.array(testGen[i][1]))

np.asarray(test_labels, dtype=int64)

#setting up a matrix
matrix = cm(yActual, yPredicted)

#plotting the heatmap
sn.heatmap(matrix, cmap='Reds', annot=True, fmt='d', xticklabels = classes, yticklabels = rclasses)

=======
hello
>>>>>>> parent of 3e79063 (first training + model complete)
