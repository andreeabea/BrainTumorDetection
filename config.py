import os
# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "brain_tumor_dataset"
# initialize the base path to the *new* directory that will contain
# the images after computing the training and testing split
BASE_PATH = "brain_tumor_new"
# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define the amount of data that will be used training
TRAIN_SPLIT = 0.65
# the amount of validation data
VAL_SPLIT = 0.25

# augmented data paths
AUG_DATASET = "augmented_dataset"
AUG_NO = os.path.sep.join([AUG_DATASET, "no"])
AUG_YES = os.path.sep.join([AUG_DATASET, "yes"])

# for cross validation
TRAIN_PATH_NO = os.path.sep.join([TRAIN_PATH, "no"])
VAL_PATH_NO = os.path.sep.join([VAL_PATH, "no"])
TRAIN_PATH_YES = os.path.sep.join([TRAIN_PATH, "yes"])
VAL_PATH_YES = os.path.sep.join([VAL_PATH, "yes"])
