from src.browser_tools import browser_action

def main():
    # Example 1: Navigate and take screenshot
    result = browser_action(
        action="goto",
        url="https://example.com"
    )
    print("Navigation result:", result["success"])
    
    # Example 2: Screenshot with analysis
    analysis = browser_action(
        action="screenshot",
        screenshot_path="page.png",
        analyze=True,
        custom_prompt="Describe the main content and any navigation elements"
    )
    print("\nPage Analysis:", analysis["data"])
    
    # Example 3: Click and type
    form_result = browser_action(
        action="type",
        selector="input[type=email]",
        text="test@example.com"
    )
    print("\nForm fill result:", form_result["success"])

if __name__ == "__main__":
    main() 