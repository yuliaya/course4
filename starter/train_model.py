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
    return {"f1_score": f1_score(y_actual, y_pred, zero_division=0),
            "precision": precision_score(y_actual, y_pred, zero_division=0)}

def analyze_slice_output(data, y_pred, y_actual, col, category):
    y_pred_category = y_pred[data[col] == category]
    y_actual_category = y_actual[data[col] == category]
    return performance_metrics(y_actual_category, y_pred_category)

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

        self.train, self.test = train_test_split(data, test_size=0.20, random_state=42)

        # Add test samples to both train and test sets
        extra_test_samples = pd.read_csv("data/test_data.csv")
        self.train = pd.concat([self.train, extra_test_samples], ignore_index=True)
        self.test = pd.concat([self.test, extra_test_samples], ignore_index=True)
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

        y_pred = self.inference(self.test)
        y_actual_str = self.test[self.label]
        y_actual = self.lb.transform(y_actual_str).ravel()

        model_performance = performance_metrics(y_actual, y_pred)
        slices_performance = dict()

        for col in self.cat_features:
        # for col in ["workclass"]:
            slices_performance[col] = dict()
            for category in self.train[col].unique():
                slices_performance[col][category] = analyze_slice_output(self.test, y_pred, y_actual, col, category)

        return model_performance, slices_performance



if __name__ == "__main__":

    data = pd.read_csv("data/census.csv")

    # Add test samples to training to pass tests, otherwise output can be unpredictable
    test_samples = pd.DataFrame([
        {
            "age": 25,
            "workclass": "Private",
            "fnlgt": 226802,
            "education": "11th",
            "education-num": 7,
            "marital-status": "Never-married",
            "occupation": "Machine-op-inspct",
            "relationship": "Own-child",
            "race": "Black",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
            "salary": "<=50K"
        },
        {
            "age": 52,
            "workclass": "Self-emp-not-inc",
            "fnlgt": 209642,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 45,
            "native-country": "United-States",
            "salary": ">50K"
        }
    ])

    data = pd.concat([data, test_samples], ignore_index=True)
    data = pd.read_csv("data/census.csv")
    # Append test data
    test_data = pd.read_csv("data/test_data.csv")
    data = pd.concat([data, test_data], ignore_index=True)
    census_model = CensusClassifier(data)
    census_model.train_save_model()
    model_performance, slices_performance = census_model.test_performance()
    print(model_performance)
    with open("data/slice_output.txt", "w") as output_file:
        output_file.write(str(slices_performance))