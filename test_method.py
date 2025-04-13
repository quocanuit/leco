import requests

url = "http://localhost:5000/judgment"

payload = {
    "question": "Tôi kết hôn năm 2020, có 1 con trai 3 tuổi. Vợ tôi không chăm con, tôi có quyền xin nuôi con không?"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    print(response.json())
else:
    print(response.text)
