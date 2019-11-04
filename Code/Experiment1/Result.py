import Utils as utils

def Result():
    score = utils.LoadObject("./Experiment1/Data/Scores.pkl")
    print('Score: ' + str(score))
