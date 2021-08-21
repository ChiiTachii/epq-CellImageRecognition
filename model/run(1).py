import model as mp
from pathlib import Path

#this is the maximum width and height that ResNet handles.
maxsize = (224, 224)

#specifies the directories for the training and testing.
trainDir = Path('./images/TRAIN')
testDir = Path('./images/TEST')

#this is a list of the 4 classes that will be used. they will be formatted into the directory.
classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

trainGen, testGen = mp.loadImages(trainDir, maxsize, testDir)

model2 = mp.loadModel('model_19082021-0245')

Continue = True

savepath = 'model_19082021-1926'

while Continue:
    model = mp.loadModel(savepath)
    mp.trainModel(model, trainGen, 20)
    if mp.evalModel(model, testGen) > mp.evalModel(model2, testGen):
        print("\nCurrent trained model beat the previous one.")
        savepath = mp.saveModel(model)
        model2 = model
    else:
        print("\nUnfortunatly, the previous model did better.")
        Continue = False
