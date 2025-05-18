import requests
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test the judgment endpoint')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    args = parser.parse_args()
    
    url = "http://localhost:5000/judgment"
    
    payload = {
        "question": "Bạn có thông tin về Bản án về ly hôn, tranh chấp nuôi con số 05/2024/HNGĐ-ST không?",
        "debug": args.debug 
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        response_data = response.json()
        
        print("Answer:", response_data['answer'])
        
        if args.debug and response_data.get('debug_info'):
            print("\n--- DEBUG INFORMATION ---")
            if 'prompt' in response_data['debug_info']:
                print(response_data['debug_info']['prompt'])
            print("--- END DEBUG INFORMATION ---\n")
    else:
        print(response.text)

if __name__ == "__main__":
    main()