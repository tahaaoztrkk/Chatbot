import requests

url = "http://127.0.0.1:8001/ask"

print("--- TEST 1: VERİ ANALİZİ (DATA) TESTİ ---")
payload_data = {
    "text": "Mobilya kategorisinden toplam kaç TL gelir elde edilmiştir?",
    "file_path": "C:/Users/tahaa/Desktop/CB/data/ornek_satislar.csv"
}
response_data = requests.post(url, json=payload_data)
if response_data.status_code == 200:
    print("Dönüş:", response_data.json())
else:
    print("HATA:", response_data.text)


print("\n-------------------------------------------------\n")

print("--- TEST 2: SOYUT BELGE/SOHBET (TEXT) TESTİ ---")
payload_text = {
    "text": "İş deneyimlerimi CV üzerinden detaylıca açıklar mısın?",
    "file_path": "C:/Users/tahaa/Desktop/CB/data/ornek_satislar.csv" 
    # TEXT testinde CSV yolunun bir önemi yok, sistem RAG deposuna gidecektir
}
response_text = requests.post(url, json=payload_text)
if response_text.status_code == 200:
    print("Dönüş:", response_text.json())
else:
    print("HATA:", response_text.text)
