import Utils as utils
import matplotlib.pyplot as plt

def Result():
    scores = utils.LoadObject("./Experiment2/Data/Scores.pkl")
    learning_rates = [0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.3, 0.5, 0.7, 1]

    xmax = learning_rates[scores.argmax()]
    ymax = scores.max()

    plt.plot(learning_rates, scores, 'Blue', zorder=1)
    dot = plt.scatter(xmax, ymax, s=None, c='Black', zorder=2)
    plt.title('Learning rate experiment')
    plt.xlabel('Learning rate')
    plt.ylabel('Score')
    plt.legend([dot], ['Best result\nLearning rate: {:.2f}\nScore: {:.2f}'.format(xmax, ymax)])

    plt.savefig('./Output/LearningRate.png')
