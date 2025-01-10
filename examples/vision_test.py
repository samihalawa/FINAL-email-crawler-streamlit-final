from src.browser_tools import browser_action
import time

def main():
    # Navigate to GitHub
    browser_action(
        action="goto",
        url="https://github.com",
        headless=False
    )
    
    # Take and analyze screenshot of sign-in button
    analysis = browser_action(
        action="screenshot",
        selector=".HeaderMenu-link--sign-in",
        screenshot_path="github_signin.png",
        analyze=True,
        custom_prompt="""Analyze this button and tell me:
1. The exact text shown
2. Button colors (background, text)
3. Size and position
4. Any hover states or animations""",
        headless=False
    )
    
    print("\nButton Analysis:", analysis["data"])
    
    # Take full page screenshot and analyze layout
    full_analysis = browser_action(
        action="screenshot",
        screenshot_path="github_full.png",
        analyze=True,
        custom_prompt="""Analyze this page layout and tell me:
1. Main navigation elements
2. Hero section content
3. Call-to-action buttons
4. Color scheme and branding""",
        headless=False
    )
    
    print("\nPage Analysis:", full_analysis["data"])

if __name__ == "__main__":
    main() 