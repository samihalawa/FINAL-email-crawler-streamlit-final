import requests
import json
import time

def trigger_automation():
    # The Streamlit server URL
    url = "http://localhost:8505"
    
    try:
        # Get the session
        session = requests.Session()
        response = session.get(url)
        
        # Navigate to automation page
        session.get(f"{url}/automation")
        
        # Simulate button click by setting session state
        data = {
            "automation_status": True,
            "automation_logs": []
        }
        session.post(f"{url}/_stcore/stream", json=data)
        
        print("Automation triggered successfully")
        
    except Exception as e:
        print(f"Error triggering automation: {str(e)}")

if __name__ == "__main__":
    trigger_automation() 