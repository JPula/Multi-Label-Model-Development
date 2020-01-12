import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

class DefaultCountVectorizer():
    def __init__(self):
        self.vectorizer = CountVectorizer()
    
    def fit_transform(self, X: pd.Series) -> pd.Series:
        return self.vectorizer.fit_transform(X)