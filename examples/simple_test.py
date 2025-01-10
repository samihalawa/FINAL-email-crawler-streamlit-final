from src.browser_tools import browser_action

def main():
    # Navigate to Google
    result = browser_action(
        action="goto",
        url="https://www.google.com",
        headless=False
    )
    print("Navigation result:", result["success"])
    
    # Take screenshot
    screenshot = browser_action(
        action="screenshot",
        screenshot_path="google.png",
        headless=False
    )
    print("\nScreenshot saved:", screenshot["success"])
    
    # Type in search box
    search = browser_action(
        action="type",
        selector="textarea[name=q]",
        text="Playwright Python",
        headless=False
    )
    print("\nSearch typed:", search["success"])

if __name__ == "__main__":
    main() 