import pandas as pd

from sklearn.naive_bayes import MultinomialNB

class DefaultMultiNB():
    def __init__(self):
        pass

    def fit(self, X: pd.Series, y: pd.DataFrame):
        self.labels = y.columns
        self.models = {}
        self.predictions = {}

        for label in self.labels:
            y_i = y[label]

            self.models[label] = MultinomialNB()
            self.models[label].fit(X, y_i)

    def predict(self, X: pd.Series) -> pd.DataFrame:
        for label in self.labels:
            y_i_pred = self.models[label].predict(X)
            
            self.predictions[label] = y_i_pred

        return pd.DataFrame(self.predictions)

    def predict_proba(self, X: pd.Series) -> pd.DataFrame:
        for label in self.labels:
            y_i_pred = self.models[label].predict_proba(X)
            
            self.predictions[label] = y_i_pred

        return pd.DataFrame(self.predictions)