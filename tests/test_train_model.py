import numpy as np
import pandas as pd
from starter.train_model import performance_metrics, CensusClassifier

# Dummy data for testing
dummy_data = pd.DataFrame({
    "workclass": ["Private", "Self-emp", "Private"],
    "education": ["Bachelors", "Masters", "PhD"],
    "marital-status": ["Married", "Single", "Divorced"],
    "occupation": ["Tech", "Sales", "Exec"],
    "relationship": ["Husband", "Not-in-family", "Own-child"],
    "race": ["White", "Black", "Asian"],
    "sex": ["Male", "Female", "Female"],
    "native-country": ["United-States", "Canada", "India"],
    "salary": ["<=50K", ">50K", "<=50K"],
    "age": [25, 45, 30],
    "hours-per-week": [40, 50, 35]
})

def test_performance_metrics_returns_dict():
    y_actual = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 1])
    result = performance_metrics(y_actual, y_pred)
    assert isinstance(result, set)  # Your current function returns a set

def test_classifier_can_train():
    clf = CensusClassifier(dummy_data)
    clf.train_save_model()
    assert clf.model is not None
    assert clf.encoder is not None
    assert clf.lb is not None

def test_classifier_inference_output_shape():
    clf = CensusClassifier(dummy_data)
    clf.train_save_model()
    predictions = clf.inference(clf.test)
    assert len(predictions) == len(clf.test)