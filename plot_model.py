import argparse

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

matplotlib.use("Agg")
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plots/plot3.png", help="path to output loss/accuracy plot")
ap.add_argument("-p2", "--plot2", type=str, default="plots/plot4.png", help="path to output loss/accuracy plot")
ap.add_argument("-p11", "--cross11", type=str, default="plots/cross11.png", help="path to output loss/accuracy plot")
ap.add_argument("-p21", "--cross21", type=str, default="plots/cross21.png", help="path to output loss/accuracy plot")
ap.add_argument("-p31", "--cross31", type=str, default="plots/cross31.png", help="path to output loss/accuracy plot")
ap.add_argument("-p41", "--cross41", type=str, default="plots/cross41.png", help="path to output loss/accuracy plot")
ap.add_argument("-p51", "--cross51", type=str, default="plots/cross51.png", help="path to output loss/accuracy plot")
ap.add_argument("-p12", "--cross12", type=str, default="plots/cross12.png", help="path to output loss/accuracy plot")
ap.add_argument("-p22", "--cross22", type=str, default="plots/cross22.png", help="path to output loss/accuracy plot")
ap.add_argument("-p32", "--cross32", type=str, default="plots/cross32.png", help="path to output loss/accuracy plot")
ap.add_argument("-p42", "--cross42", type=str, default="plots/cross42.png", help="path to output loss/accuracy plot")
ap.add_argument("-p52", "--cross52", type=str, default="plots/cross52.png", help="path to output loss/accuracy plot")
ap.add_argument("-c", "--confusionmatrix", type=str, default="plots/confusion_matrix.png", help="path to output confusion matrix plot")
args = vars(ap.parse_args())


def plot(history, H):
    # plot the training and validation accuracy
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

    # plot the training and validation loss
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


def plotConfusionMatrix(testGen, predIdxs):
    cm = metrics.confusion_matrix(testGen.classes, predIdxs)

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.savefig(args["confusionmatrix"])


# function used for cross validation plots, i representing the ith training
def plotFold(history, H, i):
    accArg = "cross" + str(i) + "1"
    lossArg = "cross" + str(i) + "2"

    # plot the training and validation accuracy
    N = len(history.history['accuracy'])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args[accArg])

    # plot the training and validation loss
    N = len(history.history['loss'])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(args[lossArg])