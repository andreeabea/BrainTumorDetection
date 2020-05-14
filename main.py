import tensorflow as tf
from imutils import paths

import config
import cv2
import numpy as np

from init import testGen

new_model = tf.keras.models.load_model('models/DenseNet80')

# Check its architecture
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(testGen, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

images = list(paths.list_images(config.TEST_PATH))

imgs = []

for image in images:
    img = cv2.imread(image)
    #cv2.imshow("LOL", img)
    img = cv2.resize(img, (75, 75))
    #img = np.reshape(img, [64, 64, 3])
    imgs.append(img)

imgs = np.array(imgs)
imgs = imgs/255
predictions = new_model.predict(imgs)

for i in range(0, len(imgs)):
    img = cv2.imread(images[i])
    cv2.imshow("Brain MRI image", img)
    #print(predictions[i])
    if predictions[i][0] < predictions[i][1]:
        print("yes")
    else:
        print("no")
    cv2.waitKey()
