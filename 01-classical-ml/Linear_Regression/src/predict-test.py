import requests

url = 'http://localhost:5000/predict'
data = {
    "surface_covered_in_m2": 100,
    "property_type": "apartment",
    "state": "Distrito Federal"
}

response = requests.post(url, json=data).json()
print(response)