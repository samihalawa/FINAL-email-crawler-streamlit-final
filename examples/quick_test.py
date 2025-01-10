from src.browser_tools import browser_action

def main():
    # Navigate to Google and analyze search box
    browser_action(
        action="goto",
        url="https://www.google.com",
        headless=False
    )
    
    # Take and analyze screenshot of search box
    analysis = browser_action(
        action="screenshot",
        selector="textarea[name=q]",
        screenshot_path="search.png",
        analyze=True,
        custom_prompt="Analyze this search box and tell me its exact position, styling, and any placeholder text",
        headless=False
    )
    
    print("\nSearch Box Analysis:", analysis["data"])

if __name__ == "__main__":
    main() 