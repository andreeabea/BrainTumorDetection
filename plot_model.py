import argparse

import matplotlib

import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plots/plotnew1.png", help="path to output loss/accuracy plot")
ap.add_argument("-p2", "--plot2", type=str, default="plots/plotnew2.png", help="path to output loss/accuracy plot")
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

