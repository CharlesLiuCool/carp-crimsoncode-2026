import requests

url = "http://127.0.0.1:8000/upload"

files = {
    "file": open("model/private_weights.pt", "rb")
}

response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Text:", response.text)