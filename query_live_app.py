import requests

url = "https://course4-app-448ec9003bbb.herokuapp.com/predict"

sample_input = {
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

response = requests.post(url, json=sample_input)

print("Status code:", response.status_code)
print("Raw response text:", response.text)

try:
    result = response.json()
    print("Prediction result:", result)
except requests.exceptions.JSONDecodeError:
    print("Failed to parse JSON. Response might be HTML error page or empty.")