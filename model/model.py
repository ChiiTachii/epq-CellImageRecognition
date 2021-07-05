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

'''
#THIS IS TOO LARGE TO PUSH TO GITHUB.
#saving the model
model.save(
        ("model\saves\{}".format(savepath)),
        include_optimizer = True,
        save_format='h5'
        )
'''
#save model as json
modJson = model.to_json()
with open(("model\saves\{}".format(savepath)), "wt") as f:
          f.write(modJson)
f.close()

#evaluating accuracy. verbose set to 2 for silence and information.
test_loss, test_acc = model.evaluate(testGen, verbose=2)
print('\nTest accuracy:', test_acc)

#loading the model at the previous datetime. When Loading, set compile to false due to using a custom optimizer that is not natively recognised.
model2 = keras.models.load_model("model\saves\model_05072021-2111", compile=False)

'''
#confusion matrix (currently does not work)
filenames = testGen.filenames
nb_samples = len(testGen)
y_prob=[]
y_act=[]
testGen.reset()
for _ in range(nb_samples):
  X_test,Y_test = testGen.next()
  y_prob.append(model.predict(X_test))
  y_act.append(Y_test)

predicted_class = [list(trainGen.class_indices.keys())[i.argmax()] for i in y_prob]
actual_class = [list(trainGen.class_indices.keys())[i.argmax()] for i in y_act]


out_df = pd.DataFrame(np.vstack([predicted_class,actual_class]).T,columns=['predicted_class','actual_class'])
confusion_matrix = pd.crosstab(out_df['actual_class'],out_df['predicted_class'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix,cmap='Blues', annot=True,fmt='d')
plt.show()
print('test accuracy : {}'.format((np.diagonal(confusion_matrix).sum()/confusion_matrix.sum().sum()*100)))
'''
=======
hello
>>>>>>> parent of 3e79063 (first training + model complete)
