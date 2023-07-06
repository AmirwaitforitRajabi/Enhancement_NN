import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib


def plot_learning_curves(loss, val_loss, path: pathlib.Path):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #plt.axis([1, 40, 0, 0.8])
    plt.ylim([0.01, 0.06])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    #hier muss man die Path angeben, damit es automatisch gespeichert wird
    plt.savefig(path.joinpath('model_loss.png'))


