from datetime import datetime

from imutils import paths
from keras import applications, Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Conv2D, BatchNormalization, MaxPooling2D

from keras.regularizers import l2

from keras.callbacks import LearningRateScheduler, EarlyStopping, History
from keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report
import numpy as np

from init import NUM_EPOCHS, INIT_LR, valGen, BS, testGen, trainGen, totalTrain, totalVal, totalTest
from plot_model import plot

import tensorflow as tf

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

#densenet 201
base_model = applications.DenseNet201(weights='imagenet', include_top=False, input_shape=[75, 75, 3])


for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[round(len(base_model.layers)*60/100):]:
    layer.trainable = True

model = Sequential()
model.add(base_model)
#activation="sigmoid"?
model.add(Conv2D(64, (2, 2), padding="valid"))
#model.add(MaxPooling2D())
#model.add(Dropout(0.5))
#model.add(Dense(2, activation='relu', kernel_regularizer=l2(0.03)))
#model.add(BatchNormalization())

#model.add(Conv2D(2, (1, 1), padding="valid"))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax", kernel_regularizer=l2(0.1)))
#sigmoid?

opt = Adam(lr=INIT_LR)#, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# define our set of callbacks and fit the model; mode =min
earlystopping = EarlyStopping(monitor='val_loss', verbose=1, patience=3)#, restore_best_weights=True)
#patience 2?
history = History()
# callbacks = [LearningRateScheduler(poly_decay), earlystopping, history]
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = [history,earlystopping]
#callbacks=[LearningRateScheduler(lr_scheduler, verbose=1)]#, earlystopping]
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    validation_steps=totalVal // BS,
    epochs=NUM_EPOCHS,
    callbacks=callbacks)

model.save(filepath="models/DenseNetNew2")

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

plot(history, H)

