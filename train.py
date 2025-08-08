import pandas as pd
from  sklearn.tree import DecisionTreeClassifier

def train_model():
    df = pd.read_csv("tv.csv")
    X = df.drop(columns=["likes"])
    y = df["likes"]

    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model