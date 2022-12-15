from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 

class CustomTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.c_[[self.get_title(x) for x in X]]

    def get_title(self, x):
        
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

        value_title = 0
        for key, value in title_mapping.items():
            if key in x:
                value_title = value
                continue
        return value_title