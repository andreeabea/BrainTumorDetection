import gc
from datetime import datetime

import cv2
import numpy
from imutils import paths
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow_core.python.keras.utils.np_utils import to_categorical

import config
from init import NUM_EPOCHS, INIT_LR, BS, testGen, totalTest, trainAug2, valAug, trainGen, valGen, valAug2
from plot_model import plotFold

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, History, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from tensorflow.keras.backend import clear_session

images = []
labels = []


def addImagesToList(inputImages,label):

    for image in inputImages:
        img = cv2.imread(image)
        img = cv2.resize(img, (75, 75))
        images.append(img)
        labels.append(label)


inputImgs = list(paths.list_images(config.TRAIN_PATH_NO))
addImagesToList(inputImgs, 0)
inputImgs = list(paths.list_images(config.VAL_PATH_NO))
addImagesToList(inputImgs, 0)
inputImgs = list(paths.list_images(config.TRAIN_PATH_YES))
addImagesToList(inputImgs, 1)
inputImgs = list(paths.list_images(config.VAL_PATH_YES))
addImagesToList(inputImgs, 1)

images = np.array(images)
labels = np.array(labels)
images = images.astype("float32") / 255.0

# one hot encoding for the class labels
#labels = to_categorical(labels, 2)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cvscores = []
i = 0

for train, valid in kfold.split(images, labels):
    i += 1
    print("Training on fold " + str(i) + "\n")
    print(str(len(images[train])) + " training images\n" + str(len(images[valid])) + " validation images\n  ")
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=[75, 75, 3])

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[round(len(base_model.layers)*60/100):]:
        layer.trainable = True

    model = Sequential()
    model.add(base_model)
    model.add(Conv2D(64, (2, 2), padding="valid", activation="sigmoid"))

    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(2, activation="softmax", kernel_regularizer=l2(0.1)))

    opt = Adam(lr=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # define our set of callbacks and fit the model
    earlystopping = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=3)
    save_best = ModelCheckpoint('models/CrossValidationModel', save_best_only=True, monitor='val_loss', mode='min')
    history = History()

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [history, tensorboard_callback, save_best, earlystopping]

    encTrainLabels = to_categorical(labels[train], 2)
    encValidLabels = to_categorical(labels[valid], 2)

    trainingGen = trainAug2.flow(images[train],
                                 encTrainLabels,
                                 shuffle=True,
                                 batch_size=BS)

    validationGen = valAug2.flow(images[valid],
                                 encValidLabels,
                                 shuffle=False,
                                 batch_size=BS)

    H = model.fit_generator(
        trainingGen,
        steps_per_epoch=len(images[train]) // BS,
        validation_data=validationGen,
        validation_steps=len(images[valid]) // BS,
        epochs=NUM_EPOCHS,
        callbacks=callbacks)

    print("[INFO] evaluating network on fold " + str(i) + " ...")
    trainingGen.reset()
    validationGen.reset()
    testGen.reset()

    # evaluate the model
    scores = model.evaluate(images[valid], encValidLabels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    predIdxs = model.predict_generator(testGen, steps=(totalTest // BS) + 1)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

    plotFold(history, H, i)

    del model
    clear_session()
    gc.collect()


cvscores = numpy.array(cvscores)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
