
# initialize the training data augmentation object
from imutils import paths
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config

# define the total number of epochs to train for along with the
# initial learning rate and batch size
NUM_EPOCHS = 100
INIT_LR = 1e-3
BS = 64

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

# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))
