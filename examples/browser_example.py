from src.browser_action import BrowserAction
import time

def main():
    # Example using context manager
    with BrowserAction(headless=False) as browser:
        # Navigate to a website
        browser.goto("https://www.example.com")
        
        # Wait for specific element
        browser.wait_for_selector("h1")
        
        # Get text content
        title = browser.get_text("h1")
        print(f"Page title: {title}")
        
        # Take screenshot
        browser.screenshot("example_screenshot.png")
        
        # Example form filling
        form_data = {
            "#username": "testuser",
            "#password": "testpass"
        }
        browser.fill_form(form_data)
        
        # Execute JavaScript
        page_height = browser.evaluate("document.body.scrollHeight")
        print(f"Page height: {page_height}px")
        
        # Small delay to see results
        time.sleep(2)

if __name__ == "__main__":
    main() 