from playwright.sync_api import sync_playwright
import time

def run_automation():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        try:
            # Navigate to Streamlit app
            print("Navigating to Streamlit app...")
            page.goto("http://localhost:8501")
            time.sleep(2)
            
            # Click Manual Search
            print("Clicking Manual Search...")
            page.click("text=Manual Search")
            time.sleep(1)
            
            # Enter search term
            print("Entering search term...")
            page.fill("input", "software developers spain")
            time.sleep(1)
            
            # Click search button
            print("Starting search...")
            page.click("text=Search")
            time.sleep(5)
            
            # Take screenshot
            print("Taking screenshot...")
            page.screenshot(path="search_results.png")
            
        except Exception as e:
            print(f"Error during automation: {str(e)}")
        finally:
            browser.close()

if __name__ == "__main__":
    print("Starting automated search...")
    run_automation() 