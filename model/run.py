import model as mp
from pathlib import Path

#this is the maximum width and height that ResNet handles.
maxsize = (224, 224)

#specifies the directories for the training and testing.
trainDir = Path('./images/TRAIN')
testDir = Path('./images/TEST_SIMPLE')

#this is a list of the 4 classes that will be used. they will be formatted into the directory.
classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

trainGen, testGen = mp.loadImages(trainDir, maxsize, testDir)

model2 = mp.loadModel('model_06072021-2225')

Continue = True

savepath = 'model_06072021-2231'

while Continue:
    model = mp.loadModel(savepath)
    mp.trainModel(model, trainGen, 10)
    if mp.evalModel(model, testGen) > mp.evalModel(model2, testGen):
        print("model beat the previous one")
        savepath = mp.saveModel(model)
        model2 = model
        cont = input("continue y/n")
        if cont != 'y':
            Continue = False
    else:
        print("the previous model did better.")
        Continue = False
