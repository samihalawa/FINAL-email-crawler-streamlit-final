from playwright.sync_api import sync_playwright
import time

def main():
    print("Starting Playwright test...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        try:
            # Navigate to app
            print("Navigating to app...")
            page.goto("http://localhost:8502", timeout=30000)
            page.screenshot(path="home.png")
            print("Took screenshot of home page")
            
            # Wait for sidebar to load and get its content
            print("Waiting for sidebar...")
            sidebar = page.wait_for_selector("div[data-testid=\"stSidebarNav\"]", timeout=30000)
            print("Sidebar content:", sidebar.inner_text())
            
            # Find all navigation links
            nav_links = page.query_selector_all("div[data-testid=\"stSidebarNav\"] a")
            print(f"Found {len(nav_links)} navigation links:")
            for link in nav_links:
                print(f"- {link.inner_text()}")
            
            # Click Manual Search in sidebar
            print("Clicking Manual Search...")
            page.click("text=Manual Search", timeout=30000)
            time.sleep(2)
            page.screenshot(path="manual_search.png")
            print("Clicked Manual Search and took screenshot")
            
            # Keep browser open for inspection
            input("Press Enter to close browser...")
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            page.screenshot(path="error.png")
            print("Took error screenshot")
        finally:
            browser.close()

if __name__ == "__main__":
    main() 