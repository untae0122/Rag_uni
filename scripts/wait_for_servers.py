
import argparse
import time
import requests
import sys

def check_server(url, timeout=5):
    try:
        # Check v1/models endpoint
        endpoint = f"{url.rstrip('/')}/models"
        response = requests.get(endpoint, timeout=timeout)
        if response.status_code == 200:
            return True
    except requests.RequestException:
        pass
    return False

def main():
    parser = argparse.ArgumentParser(description="Wait for vLLM Servers")
    parser.add_argument("--urls", nargs='+', required=True, help="List of base URLs to check (e.g. http://localhost:8000/v1)")
    parser.add_argument("--timeout", type=int, default=600, help="Max wait time in seconds")
    parser.add_argument("--interval", type=int, default=10, help="Check interval in seconds")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"[Wait] Checking servers: {args.urls}")
    
    while True:
        all_ready = True
        for url in args.urls:
            if not check_server(url):
                print(f"[Wait] {url} is NOT ready yet...")
                all_ready = False
                break # Optimization: wait for this one first
            else:
                print(f"[Wait] {url} is READY.")
        
        if all_ready:
            print("[Wait] All servers are ready!")
            sys.exit(0)
            
        if time.time() - start_time > args.timeout:
            print("[Wait] Timeout reached! Servers failed to start.")
            sys.exit(1)
            
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
