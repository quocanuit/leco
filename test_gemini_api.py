from google import genai

with open('.env', 'r') as file:
    api_key_content = file.read()

client = genai.Client(api_key=api_key_content)

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Luật hôn nhân gia đình Việt Nam"
)
print(response.text)