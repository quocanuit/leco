import requests

url = "http://localhost:5000/judgment"

payload = {
    "question": "Bạn có thông tin về Bản án về ly hôn, tranh chấp nuôi con số 05/2024/HNGĐ-ST không?"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    print(response.json())
else:
    print(response.text)