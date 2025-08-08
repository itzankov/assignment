import pandas as pd
from  sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("tv.csv")
X = df.drop(columns=["likes"])
y = df["likes"]

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Acuracy", accuracy_score(y_test, predictions))