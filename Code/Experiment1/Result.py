import Utils as utils

def Result():
    score = utils.LoadObject("./Experiment1/Data/Score.pkl")

    print('Score: ' + str(score))
