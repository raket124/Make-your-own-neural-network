import numpy as np
import Utils as utils
import matplotlib.pyplot as plt

def Result():
    results = utils.LoadObject("./Experiment5/Data/Results.pkl")

    for x in range(10):
        image = results[x].reshape((28,28))
        plt.imsave('./Output/Estimations/' + str(x) + '.png', image, cmap='gray')
