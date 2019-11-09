import numpy as np
import Utils as utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D

def Result():
    scores = utils.LoadObject("./Experiment4/Data/Scores.pkl")
    epochs = 15
    learning_rates = np.array([0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(learning_rates, range(1, epochs + 1))
    surf = ax.plot_surface(X, Y, scores, cmap=cm.coolwarm)

    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Epochs')
    ax.set_zlabel('Score')

    max_index = np.unravel_index(scores.argmax(), scores.shape)
    xmax = max_index[0] + 1
    ymax = learning_rates[max_index[1]]
    zmax = scores[max_index[0], max_index[1]]

    dot = ax.scatter(ymax, xmax, zmax, c='Black')
    plt.legend([dot], ['Best result\nLearning rate: {:.3f}\nEpoch: {:}\nScore: {:.2f}'.format(ymax, xmax, zmax)], loc='upper right')

    def Animate(i):
        ax.view_init(20, i)
        return surf

    ani = matplotlib.animation.FuncAnimation(fig, Animate, frames=360)
    ani.save('./Output/EpochAndLearningRate.gif', writer='imagemagick', fps=30)
