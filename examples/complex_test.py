from src.browser_tools import browser_action
import time

def main():
    try:
        # 1. Navigate to GitHub
        nav_result = browser_action(
            action="goto",
            url="https://github.com",
            headless=False
        )
        if not nav_result["success"]:
            raise Exception(f"Navigation failed: {nav_result['error']}")
            
        # 2. Take screenshot of sign-up section
        signup_analysis = browser_action(
            action="screenshot",
            selector=".signup-prompt",
            screenshot_path="signup.png",
            analyze=True,
            custom_prompt="""Describe this signup section in detail:
            1. All visible text and buttons
            2. Input field types and labels
            3. Button colors and styling""",
            headless=False
        )
        print("\nSignup Analysis:", signup_analysis["data"])
        
        # 3. Find and click search box
        search_result = browser_action(
            action="click",
            selector="[data-target='qbsearch-input.inputButton']",
            headless=False
        )
        if not search_result["success"]:
            raise Exception(f"Search click failed: {search_result['error']}")
            
        time.sleep(2)  # Wait for search overlay
        
        # 4. Type search query
        type_result = browser_action(
            action="type",
            selector="#query-builder-test",
            text="playwright python",
            headless=False
        )
        if not type_result["success"]:
            raise Exception(f"Type failed: {type_result['error']}")
            
        # 5. Take screenshot of search results
        results_analysis = browser_action(
            action="screenshot",
            screenshot_path="search_results.png",
            analyze=True,
            custom_prompt="""Analyze the search interface:
            1. List all visible search results
            2. Note any filters or categories
            3. Describe the layout and organization""",
            headless=False
        )
        print("\nSearch Results Analysis:", results_analysis["data"])
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main() 