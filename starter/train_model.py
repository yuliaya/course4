# Script to train machine learning model.
import os
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from starter.ml.data import process_data

# Add the necessary imports for the starter code.

# Add code to load in the data.

data = pd.read_csv("data/census.csv")
print(data.columns)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    train, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
with open("model/decision_tree.pkl", "wb") as output_file:
    pkl.dump(model, output_file)

