
import requests
import sys

BASE_URL = "https://lalityadav07-phase-detection.hf.space"

def check_url(path):
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.get(url, timeout=10)
        print(f"Checking {url} ... {resp.status_code}")
        return resp.status_code
    except Exception as e:
        print(f"Failed to connect to {url}: {e}")
        return None

def verify():
    print("=== Remote Verification ===")
    
    # 1. Check Gateway Health
    s1 = check_url("/health")
    if s1 == 200:
        print("[OK] Gateway is UP")
    else:
        print("[FAIL] Gateway is DOWN or Unreachable")

    # 2. Check API Endpoint
    s2 = check_url("/api/health")
    if s2 == 200:
        print("[OK] API is UP")
    else:
        print("[FAIL] API is DOWN or Unreachable")

    # 3. Check Streamlit Root (this is proxied)
    s3 = check_url("/")
    if s3 == 200:
        print("[OK] Streamlit Root is accessible")
    else:
        print("[FAIL] Streamlit Root is invalid")

    # 4. Check Streamlit Static Asset
    s4 = check_url("/static/js/main.js") 
    try:
        r = requests.get(f"{BASE_URL}/", timeout=10)
        if "<script" in r.text or "<noscript" in r.text:
             print("[OK] Streamlit HTML content seems valid")
        else:
             print("[WARN] Streamlit HTML content suspicious")
             print(r.text[:200])
    except:
        pass

if __name__ == "__main__":
    verify()
