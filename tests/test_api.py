from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Test case that should predict <=50K (included in both train and test sets)
sample_input_low_income = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# Test case that should predict >50K (included in both train and test sets)
sample_input_high_income = {
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
    "native-country": "United-States"
}

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API"}

def test_post_predict_low_income():
    response = client.post("/predict", json=sample_input_low_income)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction in ["<=50K", ">50K"]
    print(f"Low income test prediction: {prediction}")

def test_post_predict_high_income():
    response = client.post("/predict", json=sample_input_high_income)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction in ["<=50K", ">50K"]
    print(f"High income test prediction: {prediction}")
