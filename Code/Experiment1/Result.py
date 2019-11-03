import Utils as utils

def Result():
    scores = utils.LoadObject("./Experiment1/Data/Scores.pkl")
    print('Score: ' + str(scores))
