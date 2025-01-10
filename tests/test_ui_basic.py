from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import time

def test_manual_search_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Navigate to Streamlit app
            print("Navigating to app...")
            page.goto("http://localhost:8501")
            time.sleep(10)  # Increased initial wait time
            
            # Wait for any sidebar element
            print("Looking for sidebar...")
            page.wait_for_selector("section[data-testid='stSidebar']", timeout=20000)
            
            # Try different selectors for Manual Search
            print("Clicking Manual Search...")
            selectors = [
                "button:has-text('Manual Search')",
                "a:has-text('Manual Search')",
                "div:has-text('Manual Search')"
            ]
            
            for selector in selectors:
                try:
                    element = page.wait_for_selector(selector, timeout=5000)
                    if element:
                        element.click()
                        print(f"Successfully clicked using selector: {selector}")
                        break
                except:
                    continue
            
            time.sleep(5)  # Wait for page transition
            
            # Look for search term input with multiple selectors
            print("Looking for search term input...")
            input_selectors = [
                "div[data-testid='stTextInput'] input",
                "input[aria-label='Search Terms']",
                "input[placeholder*='search']",
                "input[type='text']"
            ]
            
            search_input = None
            for selector in input_selectors:
                try:
                    search_input = page.wait_for_selector(selector, timeout=5000)
                    if search_input:
                        print(f"Found search input using selector: {selector}")
                        break
                except:
                    continue
            
            if not search_input:
                raise Exception("Could not find search term input")
            
            print("Entering search term...")
            search_input.fill("test company")
            time.sleep(2)
            
            # Try to find the results input
            print("Looking for results input...")
            results_selectors = [
                "div[aria-label='Results per term'] input",
                "input[type='number']",
                "div[data-testid='stNumberInput'] input"
            ]
            
            for selector in results_selectors:
                try:
                    results_input = page.query_selector(selector)
                    if results_input:
                        print(f"Found results input using selector: {selector}")
                        print("Setting number of results...")
                        results_input.fill("5")
                        break
                except:
                    continue
            
            time.sleep(2)
            
            # Look for search button with multiple selectors
            print("Looking for search button...")
            button_selectors = [
                "button:has-text('Start Search')",
                "button:has-text('Search')",
                "button[kind='primary']"
            ]
            
            search_button = None
            for selector in button_selectors:
                try:
                    search_button = page.wait_for_selector(selector, timeout=5000)
                    if search_button:
                        print(f"Found search button using selector: {selector}")
                        break
                except:
                    continue
            
            if not search_button:
                raise Exception("Could not find search button")
            
            print("Clicking search button...")
            search_button.click()
            time.sleep(15)  # Increased wait time for search
            
            # Check for results or errors
            error_message = page.query_selector("div[data-testid='stAlert']")
            if error_message:
                print(f"Found error message: {error_message.text_content()}")
            
            # Take a screenshot of the final state
            page.screenshot(path="final_state.png")
            print("Test completed successfully")
            
        except PlaywrightTimeoutError as e:
            print(f"Timeout error: {str(e)}")
            page.screenshot(path="error_screenshot.png")
            raise
        except Exception as e:
            print(f"Test failed with error: {str(e)}")
            page.screenshot(path="error_screenshot.png")
            raise
        finally:
            print("Cleaning up...")
            context.close()
            browser.close()

if __name__ == "__main__":
    test_manual_search_page() 