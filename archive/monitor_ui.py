from playwright.sync_api import sync_playwright
import time
import os

def monitor_ui():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Create screenshots directory if it doesn't exist
        os.makedirs("ui_screenshots", exist_ok=True)
        
        try:
            # Navigate to the Streamlit app
            page.goto("http://localhost:8501")
            print("Connected to Streamlit app")
            
            # Initial screenshot
            page.screenshot(path="ui_screenshots/initial.png")
            print("Took initial screenshot")
            
            # Monitor for changes
            last_content = page.content()
            screenshot_count = 1
            
            while True:
                time.sleep(2)  # Check every 2 seconds
                current_content = page.content()
                
                if current_content != last_content:
                    print(f"UI change detected! Taking screenshot {screenshot_count}")
                    page.screenshot(path=f"ui_screenshots/change_{screenshot_count}.png")
                    last_content = current_content
                    screenshot_count += 1
                
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            browser.close()

if __name__ == "__main__":
    print("Starting UI monitor...")
    monitor_ui() 