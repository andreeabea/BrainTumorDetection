import argparse

from imutils import paths
from keras import applications, Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense

import matplotlib
from keras.regularizers import l2
from keras_preprocessing.image import ImageDataGenerator
import config

from keras.callbacks import LearningRateScheduler, EarlyStopping, History
from keras.optimizers import SGD
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot11.png", help="path to output loss/accuracy plot")
ap.add_argument("-p2", "--plot2", type=str, default="plot12.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# define the total number of epochs to train for along with the
# initial learning rate and batch size
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 32

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    # return the new learning rate
    return alpha

def lr_scheduler(epoch, lr):
    decay_rate = 0.85
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr


# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))


# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest")
# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(75, 75),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS)
# initialize the validation generator
valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(75, 75),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)
# initialize the testing generator
testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(75, 75),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

#densenet 201
base_model = applications.DenseNet201(weights='imagenet', include_top=False, input_shape=[75, 75, 3])

#NU
#conv2d = Conv2D(2, (1, 1), padding="same", activation="sigmoid")
#conv2d.compute_output_shape((None, 64, 64, 2048))
#x = GlobalAveragePooling2D()(x)


for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[round(len(base_model.layers)*80/100):]:
    layer.trainable = True

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
# model.add(Dense(2, activation='relu', kernel_regularizer=l2(0.01)))
#model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dense(2, activation="relu"))
#model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax", kernel_regularizer=l2(0.01)))


opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# define our set of callbacks and fit the model
earlystopping = EarlyStopping(monitor='val_accuracy',verbose=1, patience=10)#, restore_best_weights=True)
history = History()
callbacks = [LearningRateScheduler(poly_decay), earlystopping, history]
#callbacks=[LearningRateScheduler(lr_scheduler, verbose=1)]#, earlystopping]
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    validation_steps=totalVal // BS,
    epochs=NUM_EPOCHS,
    callbacks=callbacks)

model.save(filepath="models/DenseNet6")

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen, steps=(totalTest // BS) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

# plot the training loss and accuracy
N = len(history.history['accuracy'])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# plot the training loss and accuracy
N = len(history.history['loss'])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(args["plot2"])

