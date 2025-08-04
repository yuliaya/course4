from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Example input that predicts <=50K
sample_input_low_income = {
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
    "native-country": "United-States"
}

# Example input that predicts >50K
sample_input_high_income = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 287927,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States"
}

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API"}

def test_post_predict_low_income():
    response = client.post("/predict", json=sample_input_low_income)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]  # Accepts any valid label
    assert response.json()["prediction"] == "<=50K"

def test_post_predict_high_income():
    response = client.post("/predict", json=sample_input_high_income)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]
    assert response.json()["prediction"] == ">50K"