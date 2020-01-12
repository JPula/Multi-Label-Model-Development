from sklearn import metrics

class DefaultScoring():
    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred):
        accuracy = metrics.accuracy_score(y_true, y_pred)
        
        print(f'Accuracy: {accuracy}')