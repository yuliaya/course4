# Script to train machine learning model.
import pandas as pd
import pickle as pkl
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from starter.ml.data import process_data

# Add the necessary imports for the starter code.

# Add code to load in the data.


def performance_metrics(y_actual, y_pred):
    return {f1_score(y_actual, y_pred, zero_division=0), precision_score(y_actual, y_pred, zero_division=0)}

class CensusClassifier:

    def __init__(self, data):
        self.cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        self.model = None
        self.encoder = None
        self.lb = None
        self.train, self.test = train_test_split(data, test_size=0.20)
        self.label = "salary"

    def train_save_model(self):

        X_train, y_train, self.encoder, self.lb = process_data(
            self.train, categorical_features=self.cat_features, label=self.label, training=True
        )

        # Train and save a model

        self.model = DecisionTreeClassifier()
        self.model.fit(X_train, y_train)
        with open("model/decision_tree.pkl", "wb") as output_file:
            pkl.dump(self.model, output_file)
        with open("model/encoder.pkl", "wb") as output_file:
            pkl.dump(self.encoder, output_file)
        with open("model/lb.pkl", "wb") as output_file:
            pkl.dump(self.lb, output_file)

    def inference(self, data):
        X, _, _, _ = process_data(
            data, categorical_features=self.cat_features, label=self.label, training=False, encoder=self.encoder,
            lb=self.lb
        )

        return self.model.predict(X)

    def test_performance(self):
        # Proces the test data with the process_data function.

        performance_output = dict()
        y_pred = self.inference(self.test)
        y_actual_str = self.test[self.label]
        y_actual = self.lb.transform(y_actual_str).ravel()

        performance_output["model"] = performance_metrics(y_actual, y_pred)

        for col in self.cat_features:
            performance_output[col] = dict()
            for category in self.train[col].unique():
                y_pred_category = y_pred[self.test[col] == category]
                y_actual_category = y_actual[self.test[col] == category]
                performance_output[col][category] = performance_metrics(y_actual_category, y_pred_category)

        return performance_output



if __name__ == "__main__":
    data = pd.read_csv("data/census.csv")
    census_model = CensusClassifier(data)
    census_model.train_save_model()
    print(census_model.test_performance())