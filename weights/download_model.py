import os
import urllib.request
import base64

KAGGLE_USERNAME = 'ponnagaraj'
KAGGLE_KEY = input("Enter your Kaggle API Key: ")

auth_str = f"{KAGGLE_USERNAME}:{KAGGLE_KEY}"
auth_encoded = base64.b64encode(auth_str.encode()).decode()

url = "https://www.kaggleusercontent.com/kf/329835954/best_generator.pt"

headers = {
    'Authorization': f'Basic {auth_encoded}'
}

output_path = r'e:\mini project\underwater-enhancement\weights\best_generator.pt'

print(f"Downloading to: {output_path}")

try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open(output_path, 'wb') as out_file:
            chunk_size = 8192
            downloaded = 0
            total = int(response.headers.get('content-length', 0))
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = (downloaded / total) * 100
                    print(f"Progress: {percent:.1f}%")
    
    print("Download complete!")
    
except Exception as e:
    print(f"Error: {e}")
