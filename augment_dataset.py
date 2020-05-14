import os

import imutils
import numpy as np
from imutils import paths

import config
import cv2

noImages = list(paths.list_images(config.ORIG_INPUT_DATASET+"/no"))
yesImages = list(paths.list_images(config.ORIG_INPUT_DATASET+"/yes"))

# define the datasets that we'll be building
augDatasets = [
    ("no", noImages, config.AUG_NO),
    ("yes", yesImages, config.AUG_YES),
]


def fillHoles(binary):
    # Flood fill from the edges to remove edge crystals
    h, w = binary.shape[:2]
    for row in range(h):
        if binary[row, 0] == 255:
            cv2.floodFill(binary, None, (0, row), 0)
        if binary[row, w - 1] == 255:
            cv2.floodFill(binary, None, (w - 1, row), 0)

    for col in range(w):
        if binary[0, col] == 255:
            cv2.floodFill(binary, None, (col, 0), 0)
        if binary[h - 1, col] == 255:
            cv2.floodFill(binary, None, (col, h - 1), 0)

    # Cleaning up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    #binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)
    return binary


def checkArea(image, component):
    imageArea = image.size
    componentArea = np.sum(component == 255)

    if componentArea < imageArea/4:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        component = cv2.morphologyEx(component, cv2.MORPH_DILATE, kernel, iterations=100)

    return component


# finding the largest connected component and extract it (it will be the brain)
def extractBrain(image):

    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # median filtering salt and pepper noises using a 3x3 kernel matrix
    medianFiltered = cv2.medianBlur(grayscaleImage, 3)

    image = cv2.cvtColor(medianFiltered, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert it to hsv
    hsv[:, :, 2] = hsv[:, :, 2] * 1.05
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    medianFiltered2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binary using Otsu's method
    ret, thresh = cv2.threshold(medianFiltered2, 0, 255, cv2.THRESH_OTSU)

    ret, markers = cv2.connectedComponents(thresh)

    # Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
    if marker_area:
        # Get label of largest component by area
        largest_component = np.argmax(marker_area) + 1  # Add 1 since we dropped zero above
    else:
        largest_component = 1
    # Get pixels which correspond to the brain
    brain_mask = markers == largest_component

    brain_mask = np.uint8(brain_mask)

    # closing operation multiple times to get rid of holes
    kernel = np.ones((8, 8), np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    brain_mask = fillHoles(closing)

    contours, hierarchy = cv2.findContours(brain_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    brain_mask = cv2.drawContours(brain_mask, [cnt], 0, (255, 255, 255), thickness=cv2.FILLED)

    brain_mask = checkArea(thresh, brain_mask)

    image = cv2.cvtColor(medianFiltered, cv2.COLOR_GRAY2BGR)
    brain_out = image.copy()
    # In a copy of the original image, clear those pixels that don't correspond to the brain
    brain_out[brain_mask == False] = (0, 0, 0)

    cv2.floodFill(brain_out, cv2.UMat(brain_mask, cv2.CV_8UC3), (0, 0), 255)

    brain_out = cv2.drawContours(brain_out, [cnt], 0, (0, 255, 0), 3) #thickness=cv2.FILLED)

    return brain_out


# loop over the datasets
for (dType, imagePaths, baseOutput) in augDatasets:
    # show which data split we are creating
    print("[INFO] augmenting '{}' labeled images".format(dType))

    # if the output base output directory does not exist, create it
    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)

    # loop over the input image paths
    for inputPath in imagePaths:
        # extract the filename of the input image along with its
        # corresponding class label
        filename = inputPath.split(os.path.sep)[-1]

        # if the label output directory does not exist, create it
        if not os.path.exists(baseOutput):
            print("[INFO] 'creating {}' directory".format(baseOutput))
            os.makedirs(baseOutput)

        image = cv2.imread(inputPath)
        # thresholding segmentation using adaptive Gaussian thresholding
        # segmentedImage = cv2.adaptiveThreshold(medianFiltered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        preprocessedImg = extractBrain(image)

        # construct the path to the destination image and then copy
        # the image itself
        newPath = os.path.sep.join([baseOutput, filename])
        cv2.imwrite(newPath, preprocessedImg)

        # loop over the rotation angles
        for angle in np.arange(60, 300, 60):
            if angle != 180:
                rotated = imutils.rotate_bound(preprocessedImg, angle)
                newFileName = filename.split('.')[0] + 'rot' + str(angle) + '.' + filename.split('.')[1]
                newPath = os.path.sep.join([baseOutput, newFileName])
                cv2.imwrite(newPath, rotated)

        if dType is "no":
            rotated = imutils.rotate_bound(preprocessedImg, 45)
            newFileName = filename.split('.')[0] + 'rot' + str(45) + '.' + filename.split('.')[1]
            newPath = os.path.sep.join([baseOutput, newFileName])
            cv2.imwrite(newPath, rotated)

            medianFiltered = cv2.medianBlur(preprocessedImg, 3)
            newFileName = filename.split('.')[0] + 'blur.' + filename.split('.')[1]
            newPath = os.path.sep.join([baseOutput, newFileName])
            cv2.imwrite(newPath, medianFiltered)

        for i in range(-1, 2):
            flipped = cv2.flip(preprocessedImg, i)
            newFileName = filename.split('.')[0] + 'flip' + str(i) + '.' + filename.split('.')[1]
            newPath = os.path.sep.join([baseOutput, newFileName])
            cv2.imwrite(newPath, flipped)

