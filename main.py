import unicodedata

import easygui as easygui

from imutils import paths
from sklearn.metrics import classification_report

import augment_dataset
import config
import cv2
import numpy as np

from init import testGen, totalTest, BS, trainGen

from tensorflow.keras.models import load_model

from plot_model import plotConfusionMatrix

new_model = load_model('models/BestModel')
new_model.build((None, 75, 75, 3))

# check model architecture
new_model.summary()

# evaluate the restored model
print("Evaluation on validation set:")
loss, acc = new_model.evaluate(testGen, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print("[INFO] evaluating network...")
testGen.reset()
predIdxs = new_model.predict_generator(testGen, steps=(totalTest // BS) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

# plotConfusionMatrix(testGen, predIdxs)

"""
    Initial version. Making predictions on all test images.
"""
# images = list(paths.list_images(config.TEST_PATH))
#
# imgs = []
#
# for image in images:
#     img = cv2.imread(image)
#     img = cv2.resize(img, (75, 75))
#     imgs.append(img)
#
# imgs = np.array(imgs)
# imgs = imgs/255
# predictions = new_model.predict(imgs)

uni_img = easygui.fileopenbox()

while uni_img:
    img_path = unicodedata.normalize('NFKD', uni_img).encode('ascii', 'ignore')
    img_path = img_path.decode('utf-8')

    # read input image from gui
    img = cv2.imread(img_path)

    cv2.imshow("Brain MRI image", img)
    cv2.waitKey()

    # preprocess input image
    preprocessedImg = augment_dataset.extractBrain(img)

    cv2.imshow("Preprocessed image", preprocessedImg)
    cv2.waitKey()

    preprocessedImg = cv2.resize(preprocessedImg, (75, 75))
    imgs = []
    imgs.append(preprocessedImg)
    imgs = np.array(imgs)
    imgs = imgs / 255
    predictions = new_model.predict(imgs)

#for i in range(0, len(imgs)):
    print(predictions[0])
    # if the probability of the tumor to exist is greater than the probability of not
    if predictions[0][0] < predictions[0][1]:
        print("yes")
    else:
        print("no")

    uni_img = easygui.fileopenbox()


