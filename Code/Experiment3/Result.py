import Utils as utils
import matplotlib.pyplot as plt

def Result():
    scores = utils.LoadObject("./Experiment3/Data/Scores.pkl")
    epochs = [1, 2, 3, 4, 5, 7, 10, 15, 20]

    xmax = epochs[scores.argmax()]
    ymax = scores.max()

    plt.plot(epochs, scores, 'Blue', zorder=1)
    dot = plt.scatter(xmax, ymax, s=None, c='Black', zorder=2)
    plt.title('Epochs experiment')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend([dot], ['Best result\nEpoch: {:}\nScore: {:.2f}'.format(xmax, ymax)])
    plt.savefig('Output/Epoch.png')
