from playwright.sync_api import sync_playwright
import base64
import requests
import time
import os

def analyze_screenshot(image_path: str, hf_token: str, prompt: str = None) -> dict:
    """Analyze screenshot using HuggingFace Vision API"""
    with open(image_path, "rb") as img:
        img_b64 = base64.b64encode(img.read()).decode()
        
    default_prompt = f"""Analyze this screenshot and tell me:
1. What elements and text are visible
2. Their positions and styling
3. Any clickable elements or forms
<image>data:image/png;base64,{img_b64}</image>"""

    response = requests.post(
        "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf",
        headers={"Authorization": f"Bearer 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"},
        json={
            "inputs": prompt or default_prompt,
            "parameters": {"max_new_tokens": 500, "temperature": 0.2}
        }
    )
    return response.json()

def main():
    # Browser setup with inline commands
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Navigate and wait for load
        page.goto("https://example.com")
        page.wait_for_load_state("networkidle")
        
        # Take full page screenshot
        screenshot_path = "page.png"
        page.screenshot(path=screenshot_path, full_page=True)
        
        # Analyze with vision API
        hf_token = os.getenv("HUGGINGFACE_TOKEN", "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")
        analysis = analyze_screenshot(screenshot_path, hf_token)
        print("\nPage Analysis:", analysis)
        
        # Example: Find and click button based on analysis
        if "Sign in" in str(analysis):
            page.click("text=Sign in")
            time.sleep(1)
            
            # Screenshot login form
            page.screenshot(path="login.png")
            login_analysis = analyze_screenshot(
                "login.png", 
                hf_token,
                prompt="Find the username and password fields and describe their exact positions"
            )
            print("\nLogin Form Analysis:", login_analysis)
            
            # Fill form if found
            page.type("input[type=email]", "test@example.com")
            page.type("input[type=password]", "password123")
            
        browser.close()

if __name__ == "__main__":
    main() 