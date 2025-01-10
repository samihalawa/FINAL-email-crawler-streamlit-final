from src.browser_action import BrowserAction
import os
from pprint import pprint

def main():
    # Get token from environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    # Example using context manager with vision capabilities
    with BrowserAction(headless=False, hf_token=hf_token) as browser:
        # Navigate to a website
        browser.goto("https://github.com")
        
        # Example 1: Take and analyze full page screenshot
        analysis = browser.screenshot(analyze=True)
        print("\nFull page analysis:")
        pprint(analysis)
        
        # Example 2: Screenshot specific element with custom prompt
        browser.wait_for_selector(".HeaderMenu-link--sign-in")
        custom_prompt = """Analyze this button image and tell me:
        1. The exact text shown
        2. Button styling (colors, borders, etc)
        3. Position and size information
        """
        element_analysis = browser.screenshot(
            selector=".HeaderMenu-link--sign-in",
            analyze=True,
            prompt=custom_prompt
        )
        print("\nSign-in button analysis:")
        pprint(element_analysis)
        
        # Example 3: Save screenshot without analysis
        path = browser.screenshot(
            path="github_homepage.png",
            selector=".application-main"
        )
        print(f"\nScreenshot saved to: {path}")

if __name__ == "__main__":
    main() 